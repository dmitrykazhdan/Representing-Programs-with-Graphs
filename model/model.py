import tensorflow as tf
from graph_pb2 import Graph
from dpu_utils.tfmodels import SparseGGNN
from data_processing.sample_inf_processing import SampleMetaInformation, CorpusMetaInformation
import numpy as np
import os
from data_processing import graph_processing
from data_processing.graph_features import get_used_edges_type
from random import shuffle
from utils.utils import compute_f1_score


class Model:

    def __init__(self, mode, task_id, vocabulary):

        # Initialize parameter values
        self.max_node_seq_len = 32                          # Maximum number of node subtokens
        self.max_var_seq_len = 16                           # Maximum number of variable subtokens
        self.max_slots = 64                                 # Maximum number of variable occurrences
        self.batch_size = 20000                             # Number of nodes per batch sample
        self.enable_batching = True
        self.learning_rate = 0.001
        self.ggnn_params = self.get_gnn_params()
        self.vocabulary = vocabulary
        self.voc_size = len(vocabulary)
        self.slot_id = self.vocabulary.get_id_or_unk('<SLOT>')
        self.sos_token_id = self.vocabulary.get_id_or_unk('sos_token')
        self.pad_token_id = self.vocabulary.get_id_or_unk(self.vocabulary.get_pad())
        self.embedding_size = self.ggnn_params['hidden_size']
        self.ggnn_dropout = 1.0
        self.task_type = task_id

        if mode != 'train' and mode != 'infer':
            raise ValueError("Invalid mode. Please specify \'train\' or \'infer\'...")


        self.graph = tf.Graph()
        self.mode = mode

        with self.graph.as_default():

            self.placeholders = {}
            self.make_model()
            self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto())

            if self.mode == 'train':
                self.make_train_step()
                self.sess.run(tf.global_variables_initializer())


        print ("Model built successfully...")




    def get_gnn_params(self):

        gnn_params = {}
        gnn_params["n_edge_types"] = len(get_used_edges_type())
        gnn_params["hidden_size"] = 64
        gnn_params["edge_features_size"] = {}
        gnn_params["add_backwards_edges"] = True
        gnn_params["message_aggregation_type"] = "sum"
        gnn_params["layer_timesteps"] = [8]
        gnn_params["use_propagation_attention"] = False
        gnn_params["use_edge_bias"] = False
        gnn_params["graph_rnn_activation"] = "relu"
        gnn_params["graph_rnn_cell"] = "gru"
        gnn_params["residual_connections"] = {}
        gnn_params["use_edge_msg_avg_aggregation"] = False

        return gnn_params



    def make_inputs(self):

        # Node token sequences
        self.placeholders['unique_node_labels'] = tf.placeholder(name='unique_labels', shape=[None, self.max_node_seq_len], dtype=tf.int32 )
        self.placeholders['unique_node_labels_mask'] = tf.placeholder(name='unique_node_labels_mask', shape=[None, self.max_node_seq_len], dtype=tf.float32)
        self.placeholders['node_label_indices'] = tf.placeholder(name='node_label_indices', shape=[None], dtype=tf.int32)

        # Graph edge matrices
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2]) for _ in range(self.ggnn_params['n_edge_types'])]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.ggnn_params['n_edge_types']])
        self.placeholders['num_outgoing_edges_per_type'] = tf.placeholder(tf.float32, [None, self.ggnn_params['n_edge_types']])

        # Decoder sequence placeholders
        self.placeholders['decoder_targets'] = tf.placeholder(dtype=tf.int32, shape=(None, self.max_var_seq_len), name='dec_targets')
        self.placeholders['decoder_inputs'] = tf.placeholder(shape=(self.max_var_seq_len, self.placeholders['decoder_targets'].shape[0]), dtype=tf.int32, name='dec_inputs')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [self.placeholders['decoder_targets'].shape[0], self.max_var_seq_len], name='target_mask')
        self.placeholders['sos_tokens'] = tf.placeholder(shape=(self.placeholders['decoder_targets'].shape[0]), dtype=tf.int32, name='sos_tokens')
        self.placeholders['decoder_targets_length'] = tf.placeholder(shape=(self.placeholders['decoder_targets'].shape[0]), dtype=tf.int32)

        # Node identifiers of all graph nodes of the target variable
        self.placeholders['slot_ids'] = tf.placeholder(tf.int32, [self.placeholders['decoder_targets'].shape[0], self.max_slots], name='slot_ids')
        self.placeholders['slot_ids_mask'] = tf.placeholder(tf.float32, [self.placeholders['decoder_targets'].shape[0], self.max_slots], name='slot_mask')

        # Record number of graph samples in given batch (used during loss computation)
        self.placeholders['num_samples_in_batch'] = tf.placeholder(dtype=tf.float32, shape=(1), name='num_samples_in_batch')



    def get_initial_node_representation(self):

        # Compute the embedding of input node sub-tokens
        self.embedding_encoder = tf.get_variable('embedding_encoder', [self.voc_size, self.embedding_size])

        subtoken_embedding = tf.nn.embedding_lookup(params=self.embedding_encoder, ids=self.placeholders['unique_node_labels'])

        subtoken_ids_mask = tf.reshape(self.placeholders['unique_node_labels_mask'], [-1, self.max_node_seq_len, 1])

        subtoken_embedding = subtoken_ids_mask * subtoken_embedding

        unique_label_representations = tf.reduce_sum(subtoken_embedding, axis=1)

        num_subtokens = tf.reduce_sum(subtoken_ids_mask, axis=1)

        unique_label_representations /= num_subtokens

        self.node_label_representations = tf.gather(params=unique_label_representations,
                                               indices=self.placeholders['node_label_indices'])




    def make_model(self):

        # Create inputs and compute initial node representations
        self.make_inputs()
        self.get_initial_node_representation()

        # Run graph through GGNN layer
        self.gnn_model = SparseGGNN(self.ggnn_params)
        self.gnn_representation = self.gnn_model.sparse_gnn_layer(self.ggnn_dropout,
                                                        self.node_label_representations,
                                                        self.placeholders['adjacency_lists'],
                                                        self.placeholders['num_incoming_edges_per_type'],
                                                        self.placeholders['num_outgoing_edges_per_type'],
                                                        {})


        # Compute average of <SLOT> usage representations
        self.avg_representation = tf.gather(self.gnn_representation, self.placeholders['slot_ids'])
        slot_mask = tf.reshape(self.placeholders['slot_ids_mask'], [-1, self.max_slots, 1])
        slot_embedding = slot_mask * self.avg_representation
        self.avg_representation = tf.reduce_sum(slot_embedding, axis=1)
        num_slots = tf.reduce_sum(slot_mask, axis=1)
        self.avg_representation /= num_slots


        # Obtain output sequence by passing through a single GRU layer
        self.embedding_decoder = tf.get_variable('embedding_decoder', [self.voc_size, self.embedding_size])
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
        decoder_initial_state = self.avg_representation
        self.projection_layer = tf.layers.Dense(self.voc_size, use_bias=False)



        # Training
        decoder_embedding_inputs = tf.nn.embedding_lookup(self.embedding_decoder, self.placeholders['decoder_inputs'])

        self.train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedding_inputs,
                                        self.placeholders['decoder_targets_length'],
                                        time_major=True)

        self.train_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.train_helper,
                                                             initial_state=decoder_initial_state,
                                                             output_layer=self.projection_layer)

        decoder_outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(self.train_decoder)

        self.decoder_logits_train = decoder_outputs_train.rnn_output


        # Inference
        end_token = self.pad_token_id
        max_iterations = self.max_var_seq_len

        self.inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                                         start_tokens=self.placeholders['sos_tokens'],
                                                                         end_token=end_token)


        self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.inference_helper,
                                                                 initial_state=decoder_initial_state,
                                                                 output_layer=self.projection_layer)

        outputs_inference, _, _ = tf.contrib.seq2seq.dynamic_decode(self.inference_decoder,
                                                                    maximum_iterations=max_iterations)

        self.predictions = outputs_inference.sample_id




    def make_train_step(self):

        max_batch_seq_len = tf.reduce_max(self.placeholders['decoder_targets_length'])

        self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders['decoder_targets'][:, :max_batch_seq_len],
                                                                       logits=self.decoder_logits_train)

        self.train_loss = tf.reduce_sum(self.crossent * self.placeholders['target_mask'][:, :max_batch_seq_len]) / self.placeholders['num_samples_in_batch']

        # Calculate and clip gradients
        self.train_vars = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)


        grads_and_vars = self.optimizer.compute_gradients(self.train_loss, var_list=self.train_vars)

        clipped_grads = []

        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, 5.0), var))
            else:
                clipped_grads.append((grad, var))

        self.train_step = self.optimizer.apply_gradients(clipped_grads)





    # Set placeholder values using given graph input
    def create_sample(self, slot_row_id_list, node_representation, adj_lists, incoming_edges, outgoing_edges):

        # Retrieve variable token sequence
        target_token_seq = node_representation[slot_row_id_list[0]][:self.max_var_seq_len]

        # Set all occurrences of variable to <SLOT>
        slotted_node_representation = node_representation.copy()
        slotted_node_representation[slot_row_id_list, :] = self.pad_token_id
        slotted_node_representation[slot_row_id_list, 0] = self.slot_id

        node_rep_mask = (slotted_node_representation != self.pad_token_id).astype(int)

        slot_row_ids = np.zeros((1, self.max_slots))
        slot_mask = np.zeros((1, self.max_slots))
        slot_row_ids[0, 0:len(slot_row_id_list)] = slot_row_id_list
        slot_mask[0, 0:len(slot_row_id_list)] = 1

        decoder_inputs = np.zeros((self.max_var_seq_len, 1))
        decoder_targets = np.zeros((1, self.max_var_seq_len))
        target_mask = np.zeros((1, self.max_var_seq_len))
        start_tokens = np.ones((1)) * self.sos_token_id

        if self.mode == 'train':

            # Set decoder inputs and targets
            decoder_inputs = target_token_seq.copy()
            decoder_inputs = np.insert(decoder_inputs, 0, self.sos_token_id)[:-1]
            decoder_inputs = decoder_inputs.reshape(self.max_var_seq_len, 1)

            decoder_targets = target_token_seq.copy()
            decoder_targets = decoder_targets.reshape(1, self.max_var_seq_len)

            num_non_pads = np.sum(decoder_targets != self.pad_token_id) + 1
            target_mask[0, 0:num_non_pads] = 1



        # If batching is enabled, delay creation of the node representations until batch creation
        if self.enable_batching:
            unique_label_subtokens, unique_label_indices = None, None
            unique_label_inverse_indices = slotted_node_representation
        else:
            unique_label_subtokens, unique_label_indices, unique_label_inverse_indices = \
                np.unique(slotted_node_representation, return_index=True, return_inverse=True, axis=0)


        # Create the sample graph
        graph_sample = {
            self.placeholders['unique_node_labels']: unique_label_subtokens,
            self.placeholders['unique_node_labels_mask']: node_rep_mask[unique_label_indices],
            self.placeholders['node_label_indices']: unique_label_inverse_indices,
            self.placeholders['slot_ids']: slot_row_ids,
            self.placeholders['slot_ids_mask']: slot_mask,
            self.placeholders['num_incoming_edges_per_type']: incoming_edges,
            self.placeholders['num_outgoing_edges_per_type']: outgoing_edges,
            self.placeholders['decoder_targets']: decoder_targets,
            self.placeholders['decoder_inputs']: decoder_inputs,
            self.placeholders['decoder_targets_length']: np.ones((1)) * np.sum(target_mask),
            self.placeholders['sos_tokens']: start_tokens,
            self.placeholders['target_mask']: target_mask,
            self.placeholders['num_samples_in_batch']: np.ones((1))
        }

        for i in range(self.ggnn_params['n_edge_types']):
            graph_sample[self.placeholders['adjacency_lists'][i]] = adj_lists[i]

        target_name = [self.vocabulary.get_name_for_id(token_id)
                    for token_id in target_token_seq if token_id != self.pad_token_id]

        return graph_sample, target_name


    # Extract samples from given file
    def create_samples(self, filepath):

        with open(filepath, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            max_path_len = 8


            # Select sample parsing strategy depending on the specified model task
            if self.task_type == 0:
                graph_samples, slot_node_ids = graph_processing.get_usage_samples(g, max_path_len, self.max_slots,
                                                                               self.max_node_seq_len, self.pad_token_id,
                                                                               self.slot_id, self.vocabulary)

            elif self.task_type == 1:
                graph_samples, slot_node_ids = graph_processing.get_usage_samples(g, max_path_len, self.max_slots,
                                                                                  self.max_node_seq_len,
                                                                                  self.pad_token_id,
                                                                                  self.slot_id, self.vocabulary, True)

            elif self.task_type == 2:
                graph_samples, slot_node_ids = graph_processing.get_method_body_samples(g,
                                                                                  self.max_node_seq_len,
                                                                                  self.pad_token_id,
                                                                                  self.slot_id, self.vocabulary)

            else:
                raise ValueError("Invalid task id...")


            samples, labels = [], []

            for sample in graph_samples:
                new_sample, new_label = self.create_sample(*sample)
                samples.append(new_sample)
                labels.append(new_label)


            # Save sample meta-information
            samples_meta_inf = []

            for slot_node_id in slot_node_ids:
                new_inf = SampleMetaInformation(filepath, slot_node_id)
                samples_meta_inf.append(new_inf)

            return samples, labels, samples_meta_inf




    def make_batch_samples(self, graph_samples, all_labels):

        max_nodes_in_batch = self.batch_size
        batch_samples, labels = [], []
        current_batch = []
        nodes_in_curr_batch = 0

        for sample_index, graph_sample in enumerate(graph_samples):

            num_nodes_in_sample = graph_sample[self.placeholders['node_label_indices']].shape[0]

            # Skip sample if it is too big
            if num_nodes_in_sample > max_nodes_in_batch:
                continue

            # Add to current batch if there is space
            if num_nodes_in_sample + nodes_in_curr_batch < max_nodes_in_batch:
                current_batch.append(graph_sample)
                nodes_in_curr_batch += num_nodes_in_sample

            # Otherwise start creating a new batch
            else:
                batch_samples.append(self.make_batch(current_batch))
                current_batch = [graph_sample]
                nodes_in_curr_batch = num_nodes_in_sample

            labels.append(all_labels[sample_index])


        if len(current_batch) > 0:
            batch_samples.append(self.make_batch(current_batch))

        return batch_samples, labels



    # Merge set of given graph samples into a single batch
    def make_batch(self, graph_samples):

        node_offset = 0
        node_reps = []
        slot_ids, slot_masks = [], []
        num_incoming_edges_per_type, num_outgoing_edges_per_type = [], []
        decoder_targets, decoder_inputs, decoder_targets_length, decoder_masks = [], [], [], []
        adj_lists = [[] for _ in range(self.ggnn_params['n_edge_types'])]
        start_tokens = np.ones((len(graph_samples))) * self.sos_token_id

        for graph_sample in graph_samples:

            num_nodes_in_graph = graph_sample[self.placeholders['node_label_indices']].shape[0]

            node_reps.append(graph_sample[self.placeholders['node_label_indices']])

            slot_ids.append(graph_sample[self.placeholders['slot_ids']] + graph_sample[self.placeholders['slot_ids_mask']] * node_offset)

            slot_masks.append(graph_sample[self.placeholders['slot_ids_mask']])

            num_incoming_edges_per_type.append(graph_sample[self.placeholders['num_incoming_edges_per_type']])

            num_outgoing_edges_per_type.append(graph_sample[self.placeholders['num_outgoing_edges_per_type']])

            decoder_inputs.append(graph_sample[self.placeholders['decoder_inputs']])

            decoder_targets.append(graph_sample[self.placeholders['decoder_targets']])

            decoder_targets_length.append(graph_sample[self.placeholders['decoder_targets_length']])

            decoder_masks.append(graph_sample[self.placeholders['target_mask']])

            for i in range(self.ggnn_params['n_edge_types']):
                adj_lists[i].append(graph_sample[self.placeholders['adjacency_lists'][i]] + node_offset)

            node_offset += num_nodes_in_graph



        all_node_reps = np.vstack(node_reps)
        node_rep_mask = (all_node_reps != self.pad_token_id).astype(int)

        unique_label_subtokens, unique_label_indices, unique_label_inverse_indices = \
            np.unique(all_node_reps, return_index=True, return_inverse=True, axis=0)

        batch_sample = {
            self.placeholders['unique_node_labels']: unique_label_subtokens,
            self.placeholders['unique_node_labels_mask']: node_rep_mask[unique_label_indices],
            self.placeholders['node_label_indices']: unique_label_inverse_indices,
            self.placeholders['slot_ids']: np.vstack(slot_ids),
            self.placeholders['slot_ids_mask']: np.vstack(slot_masks),
            self.placeholders['num_incoming_edges_per_type']: np.vstack(num_incoming_edges_per_type),
            self.placeholders['num_outgoing_edges_per_type']: np.vstack(num_outgoing_edges_per_type),
            self.placeholders['decoder_targets']: np.vstack(decoder_targets),
            self.placeholders['decoder_inputs']: np.hstack(decoder_inputs),
            self.placeholders['decoder_targets_length']: np.hstack(decoder_targets_length),
            self.placeholders['sos_tokens']: start_tokens,
            self.placeholders['target_mask']: np.vstack(decoder_masks),
            self.placeholders['num_samples_in_batch']: np.ones((1)) * len(decoder_targets)
        }

        for i in range(self.ggnn_params['n_edge_types']):
            if len(adj_lists[i]) > 0:
                adj_list = np.vstack(adj_lists[i])
            else:
                adj_list = np.zeros((0, 2), dtype=np.int32)

            batch_sample[self.placeholders['adjacency_lists'][i]] = adj_list

        return batch_sample




    def get_samples(self, dir_path):

        graph_samples, labels, _ = self.get_samples_with_metainf(dir_path)

        return graph_samples, labels



    def get_samples_with_metainf(self, dir_path):

        graph_samples, labels, metainf = [], [], []

        n_files = sum([1 for dirpath, dirs, files in os.walk(dir_path) for filename in files if filename.endswith('proto')])
        n_processed = 0

        for dirpath, dirs, files in os.walk(dir_path):
            for filename in files:
                if filename.endswith('proto'):

                    fname = os.path.join(dirpath, filename)

                    new_samples, new_labels, new_inf = self.create_samples(fname)

                    if len(new_samples) > 0:
                        graph_samples += new_samples
                        labels += new_labels
                        metainf += new_inf

                    n_processed += 1
                    print("Processed ", n_processed/n_files * 100, "% of files...")


        zipped = list(zip(graph_samples, labels, metainf))
        shuffle(zipped)
        graph_samples, labels, metainf = zip(*zipped)

        if self.enable_batching:
            graph_samples, labels = self.make_batch_samples(graph_samples, labels)

        return graph_samples, labels, metainf



    def train(self, train_path, val_path, n_epochs, checkpoint_path):

        train_samples, train_labels = self.get_samples(train_path)
        print("Extracted training samples... ", len(train_samples))

        val_samples, val_labels, meta_inf = self.get_samples_with_metainf(val_path)
        print("Extracted validation samples... ", len(val_samples))


        with self.graph.as_default():

            for epoch in range(n_epochs):

                loss = 0

                for graph in train_samples:
                    loss += self.sess.run([self.train_loss, self.train_step], feed_dict=graph)[0]

                print("Average Epoch Loss:", (loss/len(train_samples)))
                print("Epoch: ", epoch + 1, "/", n_epochs)
                print("---------------------------------------------")


                if (epoch+1) % 5 == 0:

                    saver = tf.train.Saver()
                    saver.save(self.sess, checkpoint_path)

                    self.compute_metrics_from_graph_samples(val_samples, val_labels, meta_inf)

            saver = tf.train.Saver()
            saver.save(self.sess, checkpoint_path)






    def infer(self, corpus_path, checkpoint_path):

        test_samples, test_labels, meta_inf = self.get_samples_with_metainf(corpus_path)

        for i in range(len(test_labels)):
            meta_inf[i].true_label = test_labels[i]

        with self.graph.as_default():

            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint_path)
            print("Model loaded successfully...")

            _, _, predicted_names = self.compute_metrics_from_graph_samples(test_samples, test_labels, meta_inf)

        return test_samples, test_labels, meta_inf, predicted_names



    def get_predictions(self, graph_samples):

        predicted_names = []

        for graph in graph_samples:

            predictions = self.sess.run([self.predictions], feed_dict=graph)[0]

            for i in range(len(predictions)):

                predicted_name = [self.vocabulary.get_name_for_id(token_id) for token_id in predictions[i]]

                if self.vocabulary.get_pad() in predicted_name:
                    pad_index = predicted_name.index(self.vocabulary.get_pad())
                    predicted_name = predicted_name[:pad_index]

                predicted_names.append(predicted_name)

        return predicted_names




    def compute_metrics_from_graph_samples(self, graph_samples, test_labels, sample_infs, print_labels=False):

        predicted_names = self.get_predictions(graph_samples)
        return self.compute_metrics(predicted_names, test_labels, sample_infs, print_labels)



    # Compute F1 and accuracy scores
    def compute_metrics(self, predicted_names, test_labels, sample_infs, print_labels=False):

        n_correct, n_nonzero, f1 = 0, 0, 0

        print("Predictions: ", len(predicted_names))
        print("Test labels: ", len(test_labels))

        for i in range(len(predicted_names)):

            if print_labels:

                print("Predicted: ", [sym.encode('utf-8') for sym in predicted_names[i]])
                print("Actual: ", [sym.encode('utf-8') for sym in test_labels[i]])
                print("")
                print("")


            f1 += compute_f1_score(predicted_names[i], test_labels[i])

            if predicted_names[i] == test_labels[i]:
                n_correct += 1
                sample_infs[i].predicted_correctly = True

            else:
                sample_infs[i].predicted_correctly = False


        accuracy = n_correct / len(test_labels) * 100

        f1 = f1 * 100 / len(predicted_names)

        print("Absolute accuracy: ", accuracy)
        print("F1 score: ", f1)

        return accuracy, f1, predicted_names





    # Compute F1 and accuracy scores, as well as usage and type information
    # using the variables seen during training
    def metrics_on_seen_vars(self, train_path, test_path, checkpoint_path):

        train_samples, train_labels = self.get_samples(train_path)
        test_samples, test_labels, sample_infs, predicted_names = self.infer(test_path, checkpoint_path)

        seen_correct, seen_incorrect, unseen_correct, unseen_incorrect = 0, 0, 0, 0

        for i, sample_inf in enumerate(sample_infs):

            if test_labels[i] in train_labels:
                sample_inf.seen_in_training = True
            else:
                sample_inf.seen_in_training = False


            if test_labels[i] in train_labels and sample_inf.predicted_correctly:
                seen_correct += 1
            elif test_labels[i] in train_labels and not sample_inf.predicted_correctly:
                seen_incorrect += 1
            elif test_labels[i] not in train_labels and sample_inf.predicted_correctly:
                unseen_correct += 1
            else:
                unseen_incorrect += 1

        seen_predictions = [predicted_names[i] for i in range(len(predicted_names))
                            if sample_infs[i].seen_in_training ]

        seen_test_labels = [test_labels[i] for i in range(len(test_labels))
                            if sample_infs[i].seen_in_training ]


        seen_sample_infs = [sample_infs[i] for i in range(len(sample_infs))
                            if sample_infs[i].seen_in_training ]


        print("Metrics on seen variables: ")
        accuracy, f1, _ = self.compute_metrics(seen_predictions, seen_test_labels, seen_sample_infs)

        meta_corpus = CorpusMetaInformation(sample_infs)
        meta_corpus.process_sample_inf()












