import tensorflow as tf
from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from dpu_utils.tfmodels import SparseGGNN
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary
from typing import List, Optional, Dict, Any
from itertools import groupby
import numpy as np
from collections import defaultdict
import os




class model():

    def __init__(self):

        self.params = self.get_gnn_params()
        self.voc_size = 100
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.mode = 'train'

        with self.graph.as_default():
            self.embedding_size = self.params['hidden_size']
            self.placeholders = {}
            self.make_model()
            self.make_train_step()

            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            self.sess.run(init_op)



    # TODO: Add type embeddings
    # TODO: Add batched iteration
    # TODO: Consider other ways of handling variable-length sequences, besides padding


    def make_inputs(self):

        # Padded graph node sub-token sequences
        self.placeholders['node_token_ids'] = tf.placeholder(tf.int32, [None, self.params['hidden_size']])

        # Graph adjacency lists
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2]) for _ in range(self.params['n_edge_types'])]

        # Graph of incoming edges per type
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.params['n_edge_types']])

        # Node identifiers of all graph nodes of the target variable
        self.placeholders['slot_ids'] = tf.placeholder(tf.int32, [None], name='slot_tokens')

        #
        self.placeholders['decoder_inputs'] = tf.placeholder(shape=(64, 1), dtype=tf.int32, name='dec_inputs')

        # Actual variable name, as a padded sequence of tokens
        self.placeholders['decoder_targets'] = tf.placeholder(dtype=tf.int32, shape=(1, 64), name='dec_targets')

        # Specify output sequence lengths
        self.placeholders['decoder_targets_length'] = tf.placeholder(shape=(1), dtype=tf.int32)

        # 0/1 matrix masking out tensor elements outside of the sequence length
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [None, None])

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
                                                                        self.placeholders['num_incoming_edges_per_type'],
                                                                        {})

        # Compute average of <SLOT> usage representations
        self.placeholders['avg_representation'] = tf.expand_dims(tf.reduce_mean(tf.gather(self.placeholders['gnn_representation'],
                                                                                self.placeholders['slot_ids']), axis=0), 0)


        # Obtain output sequence by passing through a single GRU layer
        self.embedding_decoder = tf.get_variable('embedding_decoder', [self.voc_size, self.embedding_size])
        self.decoder_embedding_inputs = tf.nn.embedding_lookup(self.embedding_decoder, self.placeholders['decoder_inputs'])
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.params['hidden_size'])
        self.decoder_initial_state = self.placeholders['avg_representation']


        self.projection_layer = tf.layers.Dense(self.voc_size, use_bias=False)


        if self.mode == 'train':

            # Define training sequence decoder
            self.train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embedding_inputs,
                                                                  self.placeholders['decoder_targets_length']
                                                                  , time_major=True)

            self.train_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.train_helper,
                                                                 initial_state=self.decoder_initial_state,
                                                                 output_layer=self.projection_layer)

            self.decoder_outputs_train, e1, e2 = tf.contrib.seq2seq.dynamic_decode(self.train_decoder)
            self.decoder_logits_train = self.decoder_outputs_train.rnn_output

            print(self.decoder_logits_train.shape)


        elif self.mode == 'infer':

            # Define inference sequence decoder
            start_tokens = tf.tile(tf.constant([0], dtype=tf.int32), [self.batch_size])
            end_token = 0

            self.inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                              start_tokens=start_tokens, end_token=end_token)

            max_iterations = tf.round(tf.reduce_max(self.encoder_inputs_length)) * 2

            self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.inference_helper,
                                                                     initial_state=self.decoder_initial_state,
                                                                     output_layer=self.projection_layer)

            self.outputs_inferece, _, _ = tf.contrib.seq2seq.dynamic_decode(self.inference_decoder,
                                                                            maximum_iterations=max_iterations)

            self.predictions = self.outputs_inference.sample_id


        else:
            raise ValueError("Invalid mode. Please specify \'train\' or \'infer\'...")


        print ("Model built successfully...")

    def make_train_step(self):

        self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders['decoder_targets'], logits=self.decoder_logits_train)
        self.train_loss = tf.reduce_sum(self.placeholders['target_mask'] * self.crossent)

        # Calculate and clip gradients
        self.train_vars = tf.trainable_variables()
        self.gradients = tf.gradients(self.train_loss, self.train_vars)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 5.0)

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(0.01)
        self.train_step = self.optimizer.apply_gradients(zip(self.clipped_gradients,self.train_vars))







    def sample_train(self):

        # Sample graph
        graph_sample = {
            self.placeholders['node_token_ids']: np.random.randint(20, size=(3, 64)),
            self.placeholders['num_incoming_edges_per_type']: np.zeros((3, self.params['n_edge_types']), dtype=np.float32),
            self.placeholders['slot_ids']: np.zeros((4,), dtype=np.int32),
            self.placeholders['decoder_targets']: np.random.randint(15, size=(1, 32)),
            self.placeholders['decoder_inputs']: np.random.randint(15, size=(32, 1)),
            self.placeholders['decoder_targets_length']: np.ones((1)) * 32,
            self.placeholders['target_mask']: np.ones((32, 1))
        }

        for i in range(self.params['n_edge_types']):
            graph_sample[self.placeholders['adjacency_lists'][i]] = np.random.randint(3, size=(3, 2))


        self.train([graph_sample])
        print("Trained...")

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


    def compute_adjacency_lists(self, graph):

        id_to_ind = {}

        ind = 0
        for node in graph.node:
            id_to_ind[node.id] = ind
            ind += 1

        adj_lists = defaultdict(list)

        for edge in graph.edge:
            type_id = edge.type - 1
            adj_lists[type_id].append([id_to_ind[edge.sourceId], id_to_ind[edge.destinationId]])

        final_adj_lists = {edge_type: np.array(sorted(adj_list), dtype=np.int32)
                           for edge_type, adj_list in adj_lists.items()}

        return final_adj_lists



    def compute_edges_per_type(self, graph, incoming=True):

        # n_nodes = ... #TODO: ensure ids are consecutive

        # TODO: assumes node ids range from 0 to n_nodes. Ensure this assumption actually holds
        n_nodes, n_edge_types = 10, 5
        edges_matrix = np.zeros((n_nodes, n_edge_types), np.int64)

        if incoming:
            for e in graph.edge: edges_matrix[e.destinationId, e.type] += 1
        else:
            for e in graph.edge: edges_matrix[e.sourceId, e.type] += 1

        return edges_matrix



    def compute_initial_node_representation(self, graph):

        # Extract set of all sub-tokens and create their vocabulary
        all_tokens = []

        for n in graph.node:
            all_tokens += split_identifier_into_parts(n.contents)

        all_tokens = list(set(all_tokens))

        self.vocabulary = Vocabulary.create_vocabulary(all_tokens, max_size=100, count_threshold=0, add_unk=True)
        self.slot_id = len(self.vocabulary) + 1
        max_size = 64
        padding_element = 0



        node_representations = np.array([self.vocabulary.get_id_or_unk_multiple(split_identifier_into_parts(node.contents),
                                                                           max_size, padding_element)
                                                                            for node in graph.node])



        return node_representations



    def get_var_nodes(self, graph):

        var_nodes = {}

        id_nodes = [node.id for node in graph.node if node.type == FeatureNode.IDENTIFIER_TOKEN]


        for node in graph.node:
            if node.type == FeatureNode.SYMBOL_VAR:
                var_nodes[node.id] = []


        for edge in graph.edge:
            if edge.sourceId in var_nodes and edge.destinationId in id_nodes:
                var_nodes[edge.sourceId].append(edge.destinationId)

        return var_nodes



    def create_sample(self, filepath):

        with open(filepath, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            adj_lists = self.compute_adjacency_lists(g)
            node_reps = self.compute_initial_node_representation(g)
            variable_nodes = self.get_var_nodes(g)

            samples = []


            for var_id in variable_nodes:
                node_representation = node_reps.copy()

                for var_node in variable_nodes[var_id]:
                    node_representation[var_node, :] = 0
                    node_representation[var_node, 0] = self.slot_id


                target_mask = np.ones((self.params['hidden_size'], 1))
                var_rep = node_representation[var_node]
                for i in range(len(var_rep)):
                    if var_rep[i] == 0: target_mask[i] = 0


                graph_sample = {
                    self.placeholders['node_token_ids']: node_representation,
                    self.placeholders['num_incoming_edges_per_type']: np.zeros((node_representation.shape[0],
                                                                                self.params['n_edge_types']),
                                                                               dtype=np.float32),
                    self.placeholders['slot_ids']: variable_nodes[var_id],
                    self.placeholders['decoder_targets']: node_representation[var_node],
                    self.placeholders['decoder_inputs']: node_representation[var_node],
                    self.placeholders['decoder_targets_length']: np.ones((1)) * self.params['hidden_size'],
                    self.placeholders['target_mask']: target_mask
                }

                print(len(adj_lists))
                i = 0
                for key in adj_lists:
                    graph_sample[self.placeholders['adjacency_lists'][i]] = adj_lists[key]
                    i += 1

                samples.append(graph_sample)

            return samples





    def get_graph_samples(self, dir_path):

        graph_samples = []

        for dirpath, dirs, files in os.walk(dir_path):
            for filename in files:
                if filename[-5:] == 'proto':
                    fname = os.path.join(dirpath, filename)
                    graph_samples += self.create_sample(fname)

        return graph_samples



    def train(self, path):

        # TODO: filter out extra nodes/edges

        graph_samples = self.get_graph_samples(path)

        with self.graph.as_default():

            for iteration in range(10):
                for graph in graph_samples:
                    loss = self.sess.run([self.train_loss, self.train_step], feed_dict=graph)
                    print("Loss:", loss)



def main(path):

  m = model()
  m.train(path)


main("/Users/AdminDK/Desktop/sample_graphs")





'''
Nodes:
FAKE_AST, SYMBOL, SYMBOL_TYP, SYMBOL_VAR, SYMBOL_MTH, COMMENT_LINE, COMMENT_BLOCK, COMMENT_JAVADOC ?

Used Node Types: TOKEN, AST_ELEMENT, IDENTIFIER_TOKEN


Edges:
ASSOCIATED_TOKEN, NONE, COMMENT

Used Edge Types: NEXT_TOKEN, AST_CHILD, LAST_WRITE, LAST_USE, COMPUTED_FROM, RETURNS_TO, FORMAL_ARG_NAME, GUARDED_BY,
GUARDED_BY_NEGATION, LAST_LEXICAL_USE, 
'''






