import tensorflow as tf
from graph_pb2 import Graph
from dpu_utils.tfmodels import SparseGGNN
from data_preprocessing import SampleMetaInformation, CorpusMetaInformation
import numpy as np
import os
import graph_preprocessing
from random import shuffle



class model():

    def __init__(self, mode, vocabulary, checkpoint_path):

        # Initialize parameter values
        self.checkpoint_path = checkpoint_path
        self.max_node_seq_len = 32                          # Maximum number of node subtokens
        self.max_var_seq_len = 16                           # Maximum number of variable subtokens
        self.max_slots = 64                                 # Maximum number of variable occurrences
        self.batch_size = 8000                              # Number of nodes per batch sample
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

        if mode != 'train' and mode != 'infer':
            raise ValueError("Invalid mode. Please specify \'train\' or \'infer\'...")


        # Create model
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
        gnn_params["n_edge_types"] = len(graph_preprocessing.get_used_edges_type())
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
        self.placeholders['unique_node_labels'] = tf.placeholder(name='unique_labels',shape=[None, self.max_node_seq_len], dtype=tf.int32 )
        self.placeholders['unique_node_labels_mask'] = tf.placeholder(name='unique_node_labels_mask', shape=[None, self.max_node_seq_len], dtype=tf.float32)
        self.placeholders['node_label_indices'] = tf.placeholder(name='node_label_indices', shape=[None], dtype=tf.int32)


        # Graph adjacency lists
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2]) for _ in range(self.ggnn_params['n_edge_types'])]

        # Graph of incoming/outgoing edges per type
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.ggnn_params['n_edge_types']])
        self.placeholders['num_outgoing_edges_per_type'] = tf.placeholder(tf.float32, [None, self.ggnn_params['n_edge_types']])

        # Actual variable name, as a padded sequence of tokens
        self.placeholders['decoder_targets'] = tf.placeholder(dtype=tf.int32, shape=(None, self.max_var_seq_len), name='dec_targets')
        self.placeholders['decoder_inputs'] = tf.placeholder(shape=(self.max_var_seq_len, self.placeholders['decoder_targets'].shape[0]), dtype=tf.int32, name='dec_inputs')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [self.placeholders['decoder_targets'].shape[0], self.max_var_seq_len], name='target_mask')
        self.placeholders['sos_tokens'] = tf.placeholder(shape=(self.placeholders['decoder_targets'].shape[0]), dtype=tf.int32, name='sos_tokens')
        self.placeholders['decoder_targets_length'] = tf.placeholder(shape=(self.placeholders['decoder_targets'].shape[0]), dtype=tf.int32)

        # Node identifiers of all graph nodes of the target variable
        self.placeholders['slot_ids'] = tf.placeholder(tf.int32, [self.placeholders['decoder_targets'].shape[0], self.max_slots], name='slot_ids')
        self.placeholders['slot_ids_mask'] = tf.placeholder(tf.float32, [self.placeholders['decoder_targets'].shape[0], self.max_slots], name='slot_mask')


        self.placeholders['num_samples_in_batch'] = tf.placeholder(dtype=tf.float32, shape=(1), name='num_samples_in_batch')


    def make_initial_node_representation(self):

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

        self.make_inputs()
        self.make_initial_node_representation()

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

        # Define training sequence decoder
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
                                                          start_tokens=self.placeholders['sos_tokens'], end_token=end_token)


        self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.inference_helper,
                                                                 initial_state=decoder_initial_state,
                                                                 output_layer=self.projection_layer)

        outputs_inference, _, _ = tf.contrib.seq2seq.dynamic_decode(self.inference_decoder,
                                                                        maximum_iterations=max_iterations)

        self.predictions = outputs_inference.sample_id




    def make_train_step(self):

        self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders['decoder_targets'], logits=self.decoder_logits_train)
        self.train_loss = tf.reduce_sum(self.crossent * self.placeholders['target_mask']) / self.placeholders['num_samples_in_batch']

        # Calculate and clip gradients
        self.train_vars = tf.trainable_variables()
        self.gradients = tf.gradients(self.train_loss, self.train_vars)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 5.0)

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.train_vars))





    def create_sample(self, var_node_row_ids, node_representation, adj_lists, incoming_edges, outgoing_edges):

        # Retrieve variable token sequence
        var_token_seq = node_representation[var_node_row_ids[0]][:self.max_var_seq_len]

        # Set all occurrences of variable to <SLOT>
        slotted_node_representation = node_representation.copy()
        slotted_node_representation[var_node_row_ids, :] = self.pad_token_id
        slotted_node_representation[var_node_row_ids, 0] = self.slot_id

        node_rep_mask = (slotted_node_representation != self.pad_token_id).astype(int)

        slot_row_ids = np.zeros((1, self.max_slots))
        slot_mask = np.zeros((1, self.max_slots))
        slot_row_ids[0, 0:len(var_node_row_ids)] = var_node_row_ids
        slot_mask[0, 0:len(var_node_row_ids)] = 1

        decoder_inputs = np.zeros((self.max_var_seq_len, 1))
        decoder_targets = np.zeros((1, self.max_var_seq_len))
        target_mask = np.zeros((1, self.max_var_seq_len))

        start_tokens = np.ones((1)) * self.sos_token_id

        if self.mode == 'train':

            # Set decoder inputs and targets
            decoder_inputs = var_token_seq.copy()
            decoder_inputs = np.insert(decoder_inputs, 0, self.sos_token_id)[:-1]
            decoder_inputs = decoder_inputs.reshape(self.max_var_seq_len, 1)

            decoder_targets = var_token_seq.copy()
            decoder_targets = decoder_targets.reshape(1, self.max_var_seq_len)

            non_pads = np.sum(decoder_targets != self.pad_token_id) + 1
            target_mask[0, 0:non_pads] = 1



        # If batching is enabled, delay creation of the vocabulary until batch creation
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
            self.placeholders['decoder_targets_length']: np.ones((1)) * self.max_var_seq_len,
            self.placeholders['sos_tokens']: start_tokens,
            self.placeholders['target_mask']: target_mask,
            self.placeholders['num_samples_in_batch']: np.ones((1))
        }

        for i in range(self.ggnn_params['n_edge_types']):
            graph_sample[self.placeholders['adjacency_lists'][i]] = adj_lists[i]

        # Obtain variable name
        var_name = [self.vocabulary.get_name_for_id(token_id)
                    for token_id in var_token_seq if token_id != self.pad_token_id]

        return graph_sample, var_name




    # Generate training/test samples from a graph file
    def create_samples(self, filepath):

        with open(filepath, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            timesteps = 8
            graph_samples, sym_var_nodes = graph_preprocessing.get_method_bodies(g, timesteps, self.max_slots,
                                                                                  self.max_node_seq_len, self.pad_token_id, self.slot_id, self.vocabulary, True)

            samples, labels = [], []

            for sample in graph_samples:
                new_sample, new_label = self.create_sample(*sample)
                samples.append(new_sample)
                labels.append(new_label)


            sample_meta_inf = []

            for sym_var_node in sym_var_nodes:
                new_inf = SampleMetaInformation(filepath, sym_var_node)
                sample_meta_inf.append(new_inf)


            return samples, labels, sample_meta_inf




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

        graph_samples, labels, _ = self.get_samples_with_inf(dir_path)

        return graph_samples, labels



    def get_samples_with_inf(self, dir_path):

        graph_samples, labels, meta_sample_inf = [], [], []

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
                        meta_sample_inf += new_inf

                    n_processed += 1
                    print("Processed ", n_processed/n_files * 100, "% of files...")


        zipped = list(zip(graph_samples, labels, meta_sample_inf))
        shuffle(zipped)
        graph_samples, labels, meta_sample_inf = zip(*zipped)

        if self.enable_batching:
            graph_samples, labels = self.make_batch_samples(graph_samples, labels)

        return graph_samples, labels, meta_sample_inf




    def train(self, corpus_path, n_epochs):

        train_samples, train_labels = self.get_samples(corpus_path)

        print("Extracted samples... ", len(train_samples))

        losses = []


        with self.graph.as_default():

            for epoch in range(n_epochs):

                loss = 0

                for graph in train_samples:
                    loss += self.sess.run([self.train_loss, self.train_step], feed_dict=graph)[0]

                losses.append(loss)

                print("Average Epoch Loss:", (loss/len(train_samples)))
                print("Epoch: ", epoch + 1, "/", n_epochs)
                print("---------------------------------------------")


                # Save model every 20 epochs:
                if epoch % 20 == 0:
                    saver = tf.train.Saver()
                    saver.save(self.sess, self.checkpoint_path)


            saver = tf.train.Saver()
            saver.save(self.sess, self.checkpoint_path)




    def infer(self, corpus_path):

        test_samples, test_labels, sample_infs = self.get_samples_with_inf(corpus_path)

        with self.graph.as_default():

            saver = tf.train.Saver()
            saver.restore(self.sess, self.checkpoint_path)
            print("Model loaded successfully...")

            predicted_names = []

            offset = 0

            for graph in test_samples:

                predictions, usage_reps = self.sess.run([self.predictions, self.avg_representation], feed_dict=graph)

                for i in range(len(predictions)):

                    predicted_name = [self.vocabulary.get_name_for_id(token_id) for token_id in predictions[i]]

                    if self.vocabulary.get_pad() in predicted_name:
                        pad_index = predicted_name.index(self.vocabulary.get_pad())
                        predicted_name = predicted_name[:pad_index]

                    predicted_names.append(predicted_name)

                    sample_infs[offset].usage_rep = usage_reps[i]
                    sample_infs[offset].true_label = test_labels[offset]
                    offset += 1

            accuracy, f1 = self.process_predictions(predicted_names, test_labels, sample_infs)

            print("Absolute accuracy: ", accuracy)
            print("F1 score: ", f1)


            meta_corpus = CorpusMetaInformation(sample_infs)
            #meta_corpus.process_sample_inf()
            #meta_corpus.compute_usage_clusters()


        return test_samples, test_labels, sample_infs



    def compare_labels(self, train_path, test_path):

        train_samples, train_labels = self.get_samples(train_path)
        test_samples, test_labels, sample_infs = self.infer(test_path)

        seen_correct, seen_incorrect, unseen_correct, unseen_incorrect = 0, 0, 0, 0

        for i, sample_inf in enumerate(sample_infs):

            if test_labels[i] in train_labels and sample_inf.predicted_correctly:
                seen_correct += 1
            elif test_labels[i] in train_labels and not sample_inf.predicted_correctly:
                seen_incorrect += 1
            elif test_labels[i] not in train_labels and sample_inf.predicted_correctly:
                unseen_correct += 1
            else:
                unseen_incorrect += 1


        print("Seen, correctly predicted: ", seen_correct)
        print("Seen, incorrectly predicted: ", seen_incorrect)
        print("Unseen, predicted correctly: ", unseen_correct)
        print("Unseen, predicted incorrectly: ", unseen_incorrect)





    def compute_f1_score(self, prediction, test_label):


        pred_copy = prediction.copy()
        tp = 0

        for subtoken in set(test_label):
            if subtoken in pred_copy:
                tp += 1
                pred_copy.remove(subtoken)


        if len(prediction) > 0:
            pr = tp / len(prediction)
        else:
            pr = 0

        if len(test_label) > 0:
            rec = tp / len(test_label)
        else:
            rec = 0


        if (pr + rec) > 0:
            f1 = 2 * pr * rec / (pr + rec)
        else:
            f1 = 0

        return f1



    def process_predictions(self, predictions, test_labels, sample_infs):

        n_correct, n_nonzero, f1 = 0, 0, 0

        print("Predictions: ", len(predictions))
        print("Test labels: ", len(test_labels))

        for i in range(len(predictions)):

            print("Predicted: ", predictions[i])
            print("Actual: ", test_labels[i])
            print("")
            print("")


            f1 += self.compute_f1_score(predictions[i], test_labels[i])

            if predictions[i] == test_labels[i]:
                n_correct += 1
                sample_infs[i].predicted_correctly = True

            else:
                sample_infs[i].predicted_correctly = False


        accuracy = n_correct / len(test_labels) * 100

        f1 /= len(predictions)

        return accuracy, f1







