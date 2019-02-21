import tensorflow as tf
from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from dpu_utils.tfmodels import SparseGGNN
import numpy as np
from collections import defaultdict
import os
import vocabulary_extractor, graph_preprocessing
from random import shuffle
import math


class model():

    def __init__(self, mode, vocabulary):

        self.checkpoint_path = "/Users/AdminDK/Dropbox/Part III Modules/R252 Machine Learning " \
                               "for Programming/Project/checkpoint/train.ckpt"

        self.params = self.get_gnn_params()

        self.input_length = 16
        self.output_length = 8

        self.vocabulary = vocabulary
        self.voc_size = len(vocabulary)
        self.slot_id = self.vocabulary.get_id_or_unk('<SLOT>')
        self.sos_token = self.vocabulary.get_id_or_unk('sos_token')
        self.eos_token = self.vocabulary.get_id_or_unk('eos_token')
        self.pad_token = self.vocabulary.get_id_or_unk(self.vocabulary.get_pad())

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.batch_size = 1
        self.mode = mode

        with self.graph.as_default():
            self.embedding_size = self.params['hidden_size']
            self.placeholders = {}
            self.make_model()

            if self.mode == 'train':
                self.make_train_step()

            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            self.sess.run(init_op)



    def make_inputs(self):

        # Padded graph node sub-token sequences
        self.placeholders['node_token_ids'] = tf.placeholder(tf.int32, [None, self.input_length])

        # Graph adjacency lists
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2]) for _ in range(self.params['n_edge_types'])]

        # Graph of incoming/outgoing edges per type
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.params['n_edge_types']])
        self.placeholders['num_outgoing_edges_per_type'] = tf.placeholder(tf.float32, [None, self.params['n_edge_types']])


        # Node identifiers of all graph nodes of the target variable
        self.placeholders['slot_ids'] = tf.placeholder(tf.int32, [None, self.batch_size], name='slot_tokens')

        #
        self.placeholders['decoder_inputs'] = tf.placeholder(shape=(self.output_length, self.batch_size), dtype=tf.int32, name='dec_inputs')

        # Actual variable name, as a padded sequence of tokens
        self.placeholders['decoder_targets'] = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.output_length), name='dec_targets')

        # Specify output sequence lengths
        self.placeholders['decoder_targets_length'] = tf.placeholder(shape=(self.batch_size), dtype=tf.int32)

        # 0/1 matrix masking out tensor elements outside of the sequence length
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [None, self.batch_size])



    def make_model(self):

        self.make_inputs()

        # Compute the embedding of input node sub-tokens
        self.embedding_encoder = tf.get_variable('embedding_encoder', [self.voc_size, self.embedding_size])
        self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_encoder, self.placeholders['node_token_ids'])

        # Average the sub-token embeddings for every node
        self.placeholders['averaged_initial_representation'] = tf.reduce_mean(self.embedding_inputs, axis=1)

        # Run graph through GGNN
        self.gnn_model = SparseGGNN(self.params)
        self.placeholders['gnn_representation'] = self.gnn_model.sparse_gnn_layer(1.0,
                                                                        self.placeholders['averaged_initial_representation'],
                                                                        self.placeholders['adjacency_lists'],
                                                                        self.placeholders['num_incoming_edges_per_type'],
                                                                        self.placeholders['num_outgoing_edges_per_type'],
                                                                        {})

        # Compute average of <SLOT> usage representations
        self.placeholders['avg_representation'] = tf.expand_dims(tf.reduce_mean(tf.gather(self.placeholders['gnn_representation'],
                                                                                self.placeholders['slot_ids']), axis=0), 0)

        # Obtain output sequence by passing through a single GRU layer
        self.embedding_decoder = tf.get_variable('embedding_decoder', [self.voc_size, self.embedding_size])
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.params['hidden_size'])
        self.decoder_initial_state = self.placeholders['avg_representation']

        self.projection_layer = tf.layers.Dense(self.voc_size, use_bias=False)


        if self.mode == 'train':

            self.decoder_embedding_inputs = tf.nn.embedding_lookup(self.embedding_decoder,
                                                                   self.placeholders['decoder_inputs'])

            # Define training sequence decoder
            self.train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embedding_inputs,
                                                                  self.placeholders['decoder_targets_length']
                                                                  , time_major=True)

            self.train_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.train_helper,
                                                                 initial_state=self.decoder_initial_state,
                                                                 output_layer=self.projection_layer)

            self.decoder_outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(self.train_decoder)
            self.decoder_logits_train = self.decoder_outputs_train.rnn_output


        elif self.mode == 'infer':

            # Define inference sequence decoder
            start_tokens = tf.fill([self.batch_size], self.sos_token)
            end_token = self.pad_token
            max_iterations = self.output_length

            self.inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                              start_tokens=start_tokens, end_token=end_token)


            self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.inference_helper,
                                                                     initial_state=self.decoder_initial_state,
                                                                     output_layer=self.projection_layer)

            self.outputs_inference, _, _ = tf.contrib.seq2seq.dynamic_decode(self.inference_decoder,
                                                                            maximum_iterations=max_iterations)

            self.predictions = self.outputs_inference.sample_id


        else:
            raise ValueError("Invalid mode. Please specify \'train\' or \'infer\'...")


        print ("Model built successfully...")



    def make_train_step(self):

        self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders['decoder_targets'], logits=self.decoder_logits_train)
        self.train_loss = tf.reduce_sum(self.crossent * self.placeholders['target_mask'])


        # Calculate and clip gradients
        self.train_vars = tf.trainable_variables()
        self.gradients = tf.gradients(self.train_loss, self.train_vars)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 5.0)

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(0.0001)
        self.train_step = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.train_vars))



    def get_gnn_params(self):

        gnn_params = {}
        gnn_params["n_edge_types"] = 10
        gnn_params["hidden_size"] = 64
        gnn_params["edge_features_size"] = {}  # Dict from edge type to feature size
        gnn_params["add_backwards_edges"] = True
        gnn_params["message_aggregation_type"] = "sum"
        gnn_params["layer_timesteps"] = [8]
        gnn_params["use_propagation_attention"] = False
        gnn_params["use_edge_bias"] = False
        gnn_params["graph_rnn_activation"] = "relu"
        gnn_params["graph_rnn_cell"] = "gru"
        gnn_params["residual_connections"] = {}  #
        gnn_params["use_edge_msg_avg_aggregation"] = False

        return gnn_params




    def create_sample(self, variable_node_ids, node_representation, adj_lists, incoming_edges, outgoing_edges):

        node_rep_copy = node_representation.copy()

        # Set all occurences of variable to <SLOT>
        for variable_node_id in variable_node_ids:
            node_rep_copy[variable_node_id, :] = self.pad_token
            node_rep_copy[variable_node_id, 0] = self.slot_id


        target_mask = np.zeros((self.output_length, self.batch_size))

        variable_representation = node_representation[variable_node_ids[0]][:self.output_length]

        var_name = [self.vocabulary.get_name_for_id(token_id)
                    for token_id in variable_representation if token_id != self.pad_token]

        if self.mode == 'train':

            # Fill in target mask
            for i in range(len(variable_representation)):
                if variable_representation[i] != self.pad_token: target_mask[i, 0] = 1

            # Set decoder inputs and targets
            decoder_inputs = variable_representation.copy()
            decoder_inputs = np.insert(decoder_inputs, 0, self.sos_token)[:-1]
            decoder_inputs = decoder_inputs.reshape(self.output_length, 1)

            decoder_targets = variable_representation.copy()
            decoder_targets = decoder_targets.reshape(1, self.output_length)

        elif self.mode == 'infer':

            decoder_inputs = np.zeros((self.output_length, 1))
            decoder_targets = np.zeros((1, self.output_length))


        # Create the sample graph
        graph_sample = {
            self.placeholders['node_token_ids']: node_rep_copy,
            self.placeholders['num_incoming_edges_per_type']: incoming_edges,
            self.placeholders['num_outgoing_edges_per_type']: outgoing_edges,
            self.placeholders['slot_ids']: variable_node_ids,
            self.placeholders['decoder_targets']: decoder_targets,
            self.placeholders['decoder_inputs']: decoder_inputs,
            self.placeholders['decoder_targets_length']: np.ones((1)) * self.output_length,
            self.placeholders['target_mask']: target_mask
        }

        i = 0
        for key in adj_lists:
            graph_sample[self.placeholders['adjacency_lists'][i]] = adj_lists[key]
            i += 1


        return graph_sample, var_name



    # Generate training/test samples from a graph file
    def create_samples(self, filepath):

        with open(filepath, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            filtered_nodes, filtered_edges = graph_preprocessing.filter_graph(g)

            id_to_index_map = graph_preprocessing.get_node_id_to_index_map(filtered_nodes)

            variable_node_ids = graph_preprocessing.get_var_nodes_map(g, id_to_index_map)

            adjacency_lists = graph_preprocessing.compute_adjacency_lists(filtered_edges, id_to_index_map)

            node_representations = graph_preprocessing.compute_initial_node_representation(filtered_nodes, self.input_length,
                                                                                           self.pad_token, self.vocabulary)

            incoming_edges_per_type, outgoing_edges_per_type = \
                graph_preprocessing.compute_edges_per_type(len(node_representations), adjacency_lists)


            samples, labels = [], []

            for variable_root_id in variable_node_ids:

                new_sample, new_label = self.create_sample(variable_node_ids[variable_root_id],
                                                           node_representations, adjacency_lists,
                                                           incoming_edges_per_type, outgoing_edges_per_type)

                samples.append(new_sample)
                labels.append(new_label)

            return samples, labels




    def make_batch_samples(self, graph_samples):

        batch_samples = []
        shuffle(graph_samples)

        n_batches = math.ceil(len(graph_samples)/self.batch_size)

        for i in range(n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(graph_samples))
            batch_samples.append(self.make_batch(graph_samples[start:end]))


        return batch_samples




    def make_batch(self, graph_samples):

        node_offset = 0
        node_representations = []
        adj_lists = [[] for _ in range(self.params['n_edge_types'])]
        num_incoming_edges_per_type = []
        num_outgoing_edges_per_type = []
        decoder_inputs = []
        decoder_targets = []
        decoder_targets_length = []
        decoder_masks = []

        for graph_sample in graph_samples:

            num_nodes_in_graph = len(graph_sample[self.placeholders['node_token_ids']])
            node_representations.extend(graph_sample[self.placeholders['node_token_ids']])


            for i in range(self.params['n_edge_types']):
                if i in graph_sample[self.placeholders['adjacency_lists']]:
                    adj_lists[i].append(graph_sample[self.placeholders['adjacency_lists'][i]] + node_offset)

            num_incoming_edges_per_type.append(self.placeholders['num_incoming_edges_per_type'])
            num_outgoing_edges_per_type.append(self.placeholders['num_outgoing_edges_per_type'])
            decoder_inputs.append(self.placeholders['decoder_inputs'])
            decoder_targets.append(self.placeholders['decoder_targets'])
            decoder_targets_length.append(self.placeholders['decoder_targets_length'])
            decoder_masks.append(self.placeholders['target_mask'])

            # TODO: consider how to handle slot ids

            node_offset += num_nodes_in_graph


        batch_sample = {
            self.placeholders['node_token_ids']: np.vstack(node_representations),
            self.placeholders['num_incoming_edges_per_type']: np.vstack(num_incoming_edges_per_type),
            self.placeholders['num_outgoing_edges_per_type']: np.vstack(num_outgoing_edges_per_type),
            self.placeholders['slot_ids']: None,# TBC

            self.placeholders['decoder_targets']: np.vstack(decoder_targets),
            self.placeholders['decoder_inputs']: np.hstack(decoder_inputs),
            self.placeholders['decoder_targets_length']: np.hstack(decoder_targets_length),
            self.placeholders['target_mask']: np.hstack(decoder_masks)
        }


        for i in range(self.params['n_edge_types']):
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
                    new_samples, new_labels = self.create_samples(fname)

                    graph_samples += new_samples
                    labels += new_labels

        return graph_samples, labels



    def train(self, corpus_path):

        n_epochs = 150
        train_samples, _ = self.get_samples(corpus_path)

        print("Train vals: ", _)

        with self.graph.as_default():

            for epoch in range(n_epochs):

                loss = 0

                for graph in train_samples:
                    loss += self.sess.run([self.train_loss, self.train_step], feed_dict=graph)[0]

                print("Average Epoch Loss:", (loss/len(train_samples)))
                print("Epoch: ", epoch)
                print("---------------------------------------------")

            saver = tf.train.Saver()
            saver.save(self.sess, self.checkpoint_path)



    def infer(self, corpus_path):

        test_samples, test_labels = self.get_samples(corpus_path)

        print("Test vals: ", test_labels)

        with self.graph.as_default():

            saver = tf.train.Saver()
            saver.restore(self.sess, self.checkpoint_path)
            print("Model loaded successfully...")

            n_correct = 0

            for i, graph in enumerate(test_samples):

                predictions = self.sess.run([self.predictions], feed_dict=graph)[0]

                predicted_name = [self.vocabulary.get_name_for_id(token_id) for token_id in predictions[0]]

                print("Predicted: ", predicted_name)
                print("Actual: ", test_labels[i])
                print("")
                print("")

                if predicted_name[:-1] == test_labels[i]: n_correct += 1


            print("Absolute accuracy: ", n_correct/len(test_samples) * 100)









def main():

  # Training:
  vocabulary = vocabulary_extractor.create_vocabulary_from_corpus(train_path, token_path)
  m = model('train', vocabulary)
  m.train(train_path)


  # Inference
  vocabulary = vocabulary_extractor.load_vocabulary(token_path)
  m = model('infer', vocabulary)
  m.infer(test_path)







train_path = "/Users/AdminDK/Desktop/train_graphs"
test_path = "/Users/AdminDK/Desktop/test_graphs"
token_path = "/Users/AdminDK/Desktop/tokens.txt"

main()






