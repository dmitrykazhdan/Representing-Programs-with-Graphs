import tensorflow as tf
from graph_pb2 import Graph
from dpu_utils.tfmodels import SparseGGNN
import numpy as np
import os
import graph_preprocessing
from random import shuffle
import math



class model():

    def __init__(self, mode, vocabulary, checkpoint_path):

        # Initialize parameter values
        self.checkpoint_path = checkpoint_path
        self.max_node_seq_len = 16
        self.max_var_seq_len = 8
        self.max_slots = 64
        self.batch_size = 64
        self.learning_rate = 0.001
        self.ggnn_dropout = 0.9
        self.ggnn_params = self.get_gnn_params()
        self.vocabulary = vocabulary
        self.voc_size = len(vocabulary)
        self.slot_id = self.vocabulary.get_id_or_unk('<SLOT>')
        self.sos_token = self.vocabulary.get_id_or_unk('sos_token')
        self.pad_token = self.vocabulary.get_id_or_unk(self.vocabulary.get_pad())
        self.embedding_size = self.ggnn_params['hidden_size']

        if mode != 'train' and mode != 'infer':
            raise ValueError("Invalid mode. Please specify \'train\' or \'infer\'...")


        # Create model
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.mode = mode

        with self.graph.as_default():

            self.placeholders = {}
            self.make_model()

            if self.mode == 'train':
                self.make_train_step()

            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            self.sess.run(init_op)


        print ("Model built successfully...")



    def get_gnn_params(self):

        gnn_params = {}
        gnn_params["n_edge_types"] = 10
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
        self.placeholders['unique_node_labels'] = tf.placeholder(name='unique_labels',shape=[None, self.max_node_seq_len],dtype=tf.int32 )
        self.placeholders['unique_node_labels_mask'] = tf.placeholder(name='unique_node_labels_mask',shape=[None, self.max_node_seq_len],dtype=tf.float32)
        self.placeholders['node_label_indices'] = tf.placeholder(name='node_label_indices', shape=[None], dtype=tf.int32)


        # Graph adjacency lists
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2]) for _ in range(self.ggnn_params['n_edge_types'])]

        # Graph of incoming/outgoing edges per type
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.ggnn_params['n_edge_types']])
        self.placeholders['num_outgoing_edges_per_type'] = tf.placeholder(tf.float32, [None, self.ggnn_params['n_edge_types']])


        # Node identifiers of all graph nodes of the target variable
        self.placeholders['slot_ids'] = tf.placeholder(tf.int32, [self.batch_size, self.max_slots], name='slot_ids')
        self.placeholders['slot_ids_mask'] = tf.placeholder(tf.float32, [self.batch_size, self.max_slots], name='slot_mask')

        # Actual variable name, as a padded sequence of tokens
        self.placeholders['decoder_inputs'] = tf.placeholder(shape=(self.max_var_seq_len, self.batch_size), dtype=tf.int32, name='dec_inputs')
        self.placeholders['decoder_targets'] = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.max_var_seq_len), name='dec_targets')

        # Specify output sequence lengths
        self.placeholders['decoder_targets_length'] = tf.placeholder(shape=(self.batch_size), dtype=tf.int32)

        # 0/1 matrix masking out tensor elements outside of the sequence length
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [self.batch_size, self.max_var_seq_len], name='target_mask')




    def make_initial_node_representation(self):

        # Compute the embedding of input node sub-tokens
        embedding_encoder = tf.get_variable('embedding_encoder', [self.voc_size, self.embedding_size])

        subtoken_embedding = tf.nn.embedding_lookup(params=embedding_encoder, ids=self.placeholders['unique_node_labels'])

        subtoken_ids_mask = tf.reshape(self.placeholders['unique_node_labels_mask'], [-1, self.max_node_seq_len, 1])

        subtoken_embedding = subtoken_ids_mask * subtoken_embedding

        unique_label_representations = tf.reduce_sum(subtoken_embedding, axis=1)

        num_subtokens = tf.reduce_sum(subtoken_ids_mask, axis=1)

        unique_label_representations /= num_subtokens

        node_label_representations = tf.gather(params=unique_label_representations,
                                               indices=self.placeholders['node_label_indices'])

        return node_label_representations



    def make_model(self):

        self.make_inputs()

        # Get initial embeddings for every node
        initial_representation = self.make_initial_node_representation()

        # Run graph through GGNN layer
        gnn_model = SparseGGNN(self.ggnn_params)
        gnn_representation = gnn_model.sparse_gnn_layer(self.ggnn_dropout,
                                                        initial_representation,
                                                        self.placeholders['adjacency_lists'],
                                                        self.placeholders['num_incoming_edges_per_type'],
                                                        self.placeholders['num_outgoing_edges_per_type'],
                                                        {})


        # Compute average of <SLOT> usage representations
        avg_representation = tf.gather(gnn_representation, self.placeholders['slot_ids'])
        slot_mask = tf.reshape(self.placeholders['slot_ids_mask'], [-1, self.max_slots, 1])
        slot_embedding = slot_mask * avg_representation
        avg_representation = tf.reduce_sum(slot_embedding, axis=1)
        num_slots = tf.reduce_sum(slot_mask, axis=1)
        avg_representation /= num_slots


        # Obtain output sequence by passing through a single GRU layer
        embedding_decoder = tf.get_variable('embedding_decoder', [self.voc_size, self.embedding_size])
        decoder_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
        decoder_initial_state = avg_representation
        projection_layer = tf.layers.Dense(self.voc_size, use_bias=False)


        if self.mode == 'train':

            decoder_embedding_inputs = tf.nn.embedding_lookup(embedding_decoder, self.placeholders['decoder_inputs'])

            # Define training sequence decoder
            train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedding_inputs,
                                                             self.placeholders['decoder_targets_length']
                                                                  , time_major=True)

            train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper,
                                                                 initial_state=decoder_initial_state,
                                                                 output_layer=projection_layer)

            decoder_outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)

            self.decoder_logits_train = decoder_outputs_train.rnn_output



        elif self.mode == 'infer':

            # Define inference sequence decoder
            start_tokens = tf.fill([self.batch_size], self.sos_token)
            end_token = self.pad_token
            max_iterations = self.max_var_seq_len

            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder,
                                                              start_tokens=start_tokens, end_token=end_token)


            inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, inference_helper,
                                                                     initial_state=decoder_initial_state,
                                                                     output_layer=projection_layer)

            outputs_inference, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                            maximum_iterations=max_iterations)

            self.predictions = outputs_inference.sample_id



    def make_train_step(self):

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders['decoder_targets'], logits=self.decoder_logits_train)
        self.train_loss = tf.reduce_sum(tf.multiply(crossent, self.placeholders['target_mask']))

        # Calculate and clip gradients
        train_vars = tf.trainable_variables()
        gradients = tf.gradients(self.train_loss, train_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = self.optimizer.apply_gradients(zip(clipped_gradients, train_vars))





    def create_sample(self, variable_node_ids, node_representation, adj_lists, incoming_edges, outgoing_edges):

        # Retrieve variable token sequence
        var_token_seq = node_representation[variable_node_ids[0]][:self.max_var_seq_len]

        # Set all occurrences of variable to <SLOT>
        slotted_node_representation = node_representation.copy()
        slotted_node_representation[variable_node_ids, :] = self.pad_token
        slotted_node_representation[variable_node_ids, 0] = self.slot_id

        node_rep_mask = slotted_node_representation != self.pad_token

        print("Copied node rep")

        unique_label_subtokens, node_label_indices, unique_label_inverse_indices = \
            np.unique(slotted_node_representation, return_index=True, return_inverse=True, axis=0)

        print("Computing unique labels")

        slot_ids = np.zeros((1, self.max_slots))
        slot_ids[0, :len(variable_node_ids)] = variable_node_ids
        slot_mask = slot_ids != 0

        target_mask = np.zeros((1, self.max_var_seq_len))

        if self.mode == 'train':

            # Set decoder inputs and targets
            decoder_inputs = var_token_seq.copy()
            decoder_inputs = np.insert(decoder_inputs, 0, self.sos_token)[:-1]
            decoder_inputs = decoder_inputs.reshape(self.max_var_seq_len, 1)

            decoder_targets = var_token_seq.copy()
            decoder_targets = decoder_targets.reshape(1, self.max_var_seq_len)

            non_pads = np.sum(decoder_targets != self.pad_token) + 1
            target_mask[0, 0:non_pads] = 1

        elif self.mode == 'infer':

            decoder_inputs = np.zeros((self.max_var_seq_len, 1))
            decoder_targets = np.zeros((1, self.max_var_seq_len))


        # Create the sample graph
        graph_sample = {
            self.placeholders['unique_node_labels']: unique_label_subtokens,
            self.placeholders['unique_node_labels_mask']: node_rep_mask[node_label_indices],
            self.placeholders['node_label_indices']: unique_label_inverse_indices,
            self.placeholders['slot_ids']: slot_ids,
            self.placeholders['slot_ids_mask']: slot_mask,
            self.placeholders['num_incoming_edges_per_type']: incoming_edges,
            self.placeholders['num_outgoing_edges_per_type']: outgoing_edges,
            self.placeholders['decoder_targets']: decoder_targets,
            self.placeholders['decoder_inputs']: decoder_inputs,
            self.placeholders['decoder_targets_length']: np.ones((1)) * self.max_var_seq_len,
            self.placeholders['target_mask']: target_mask
        }

        i = 0
        for key in adj_lists:
            graph_sample[self.placeholders['adjacency_lists'][i]] = adj_lists[key]
            i += 1


        # Obtain variable name
        var_name = [self.vocabulary.get_name_for_id(token_id)
                    for token_id in var_token_seq if token_id != self.pad_token]

        return graph_sample, var_name




    # Generate training/test samples from a graph file
    def create_samples(self, filepath):

        with open(filepath, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            variable_node_ids, node_representations, adjacency_lists, \
            incoming_edges_per_type, outgoing_edges_per_type = graph_preprocessing.compute_sample_data(g, self.max_node_seq_len,
                                                                                           self.pad_token, self.vocabulary)

            print("Pre-processed graph")

            samples, labels = [], []


            var_ids = list(variable_node_ids.keys())
            shuffle(var_ids)


            for var_root_node_id in var_ids:

                new_sample, new_label = self.create_sample(variable_node_ids[var_root_node_id],
                                                           node_representations, adjacency_lists,
                                                           incoming_edges_per_type, outgoing_edges_per_type)

                samples.append(new_sample)
                labels.append(new_label)

                print("Size: ", new_sample[self.placeholders['node_label_indices']].shape[0])

            return samples, labels




    def make_batch_samples(self, graph_samples, labels):

        zipped = list(zip(graph_samples, labels))
        shuffle(zipped)
        graph_samples, all_labels = zip(*zipped)

        batch_samples, labels = [], []

        n_batches = math.ceil(len(graph_samples)/self.batch_size)

        for i in range(n_batches - 1):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(graph_samples))
            new_batch = self.make_batch(graph_samples[start:end])
            batch_samples.append(new_batch)
            labels += all_labels[start:end]

        return batch_samples, labels



    # Merge set of given graph samples into a single batch
    def make_batch(self, graph_samples):

        node_offset = 0
        unique_labels, label_masks, label_indices = [], [], []
        slot_ids, slot_masks = [], []
        num_incoming_edges_per_type, num_outgoing_edges_per_type = [], []
        decoder_targets, decoder_inputs, decoder_targets_length, decoder_masks = [], [], [], []
        adj_lists = [[] for _ in range(self.ggnn_params['n_edge_types'])]

        for graph_sample in graph_samples:

            num_nodes_in_graph = len(graph_sample[self.placeholders['node_label_indices']])

            label_indices.append(graph_sample[self.placeholders['unique_node_labels']])
            label_masks.append(graph_sample[self.placeholders['unique_node_labels_mask']])
            unique_labels.append(graph_sample[self.placeholders['node_label_indices']])
            slot_ids.append(graph_sample[self.placeholders['slot_ids']])
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


        batch_sample = {
            self.placeholders['unique_node_labels']: np.vstack(label_indices),
            self.placeholders['unique_node_labels_mask']: np.vstack(label_masks),
            self.placeholders['node_label_indices']: np.hstack(unique_labels),
            self.placeholders['slot_ids']: np.vstack(slot_ids),
            self.placeholders['slot_ids_mask']: np.vstack(slot_masks),
            self.placeholders['num_incoming_edges_per_type']: np.vstack(num_incoming_edges_per_type),
            self.placeholders['num_outgoing_edges_per_type']: np.vstack(num_outgoing_edges_per_type),
            self.placeholders['decoder_targets']: np.vstack(decoder_targets),
            self.placeholders['decoder_inputs']: np.hstack(decoder_inputs),
            self.placeholders['decoder_targets_length']: np.hstack(decoder_targets_length),
            self.placeholders['target_mask']: np.vstack(decoder_masks)
        }

        for i in range(self.ggnn_params['n_edge_types']):
            if len(adj_lists[i]) > 0:
                adj_list = np.concatenate(adj_lists[i])
            else:
                adj_list = np.zeros((0, 2), dtype=np.int32)

            batch_sample[self.placeholders['adjacency_lists'][i]] = adj_list

        return batch_sample



    def get_samples(self, dir_path):

        graph_samples, labels = [], []

        for dirpath, dirs, files in os.walk(dir_path):
            for filename in files:
                if filename[-5:] == 'proto':
                    fname = os.path.join(dirpath, filename)

                    f_size = os.path.getsize(fname)/1000

                    if f_size > 100 and f_size < 400:
                        new_samples, new_labels = self.create_samples(fname)

                        graph_samples += new_samples
                        labels += new_labels

                    print("Processed new file...")

        return graph_samples, labels



    def train(self, corpus_path, n_epochs):

        train_samples, _ = self.get_samples(corpus_path)

        print("Obtained samples...")

        train_samples, _ = self.make_batch_samples(train_samples, _)

        print("Obtained batches...")

        losses = []

        print("Train vals: ", _)

        with self.graph.as_default():

            for epoch in range(n_epochs):

                loss = 0

                for graph in train_samples:
                    loss += self.sess.run([self.train_loss, self.train_step], feed_dict=graph)[0]

                losses.append(loss)

                print("Average Epoch Loss:", (loss/len(train_samples)))
                print("Epoch: ", epoch)
                print("---------------------------------------------")

            saver = tf.train.Saver()
            saver.save(self.sess, self.checkpoint_path)



    def process_predictions(selfs, predictions, test_labels):

        n_correct = 0

        for i in range(len(predictions)):

            print("Predicted: ", predictions[i])
            print("Actual: ", test_labels[i])
            print("")
            print("")

            if predictions[i] == test_labels[i]: n_correct += 1

        accuracy = n_correct / len(test_labels) * 100

        return accuracy




    def infer(self, corpus_path):

        test_samples, test_labels = self.get_samples(corpus_path)
        test_samples, test_labels = self.make_batch_samples(test_samples, test_labels)

        print("Test vals: ", test_labels)

        with self.graph.as_default():

            saver = tf.train.Saver()
            saver.restore(self.sess, self.checkpoint_path)
            print("Model loaded successfully...")

            predicted_names = []

            for graph in test_samples:

                predictions = self.sess.run([self.predictions], feed_dict=graph)[0]

                for i in range(self.batch_size):

                    predicted_name = [self.vocabulary.get_name_for_id(token_id) for token_id in predictions[i]]

                    if self.vocabulary.get_pad() in predicted_name:
                        pad_index = predicted_name.index(self.vocabulary.get_pad())
                        predicted_name = predicted_name[:pad_index]

                    predicted_names.append(predicted_name)


            accuracy = self.process_predictions(predicted_names, test_labels)

            print("Absolute accuracy: ", accuracy)













