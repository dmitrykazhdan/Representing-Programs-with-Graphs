import tensorflow as tf
from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from dpu_utils.tfmodels import SparseGGNN
from typing import List, Optional, Dict, Any
from itertools import groupby
import numpy as np
import os




class model():

    def __init__(self):

        self.params = get_gnn_params()
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
        self.decoder_embedding_inputs = tf.nn.embedding_lookup(self.embedding_decoder, self.placeholders['decoder_targets'])
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

            self.decoder_outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(self.train_decoder)
            self.decoder_logits_train = self.decoder_outputs_train.rnn_output


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



    def make_inputs(self):

        # Padded graph node sub-token sequences
        self.placeholders['node_token_ids'] = tf.placeholder(tf.int32, [None, self.params['hidden_size']])

        # Graph adjacency lists
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2]) for _ in range(self.params['n_edge_types'])]

        # Graph of incoming edges per type
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.params['n_edge_types']])

        # Node identifiers of all graph nodes of the target variable
        self.placeholders['slot_ids'] = tf.placeholder(tf.int32, [None], name='slot_tokens')

        # Actual variable name, as a padded sequence of tokens
        self.placeholders['decoder_targets'] = tf.placeholder(tf.int32, [None, None], name='train_label')

        # Specify output sequence lengths
        self.placeholders['decoder_targets_length'] = tf.placeholder(shape=(None,), dtype=tf.int32)

        # 0/1 matrix masking out tensor elements outside of the sequence length
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [None, None])




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




    def train(self, graph_samples):

        with self.graph.as_default():

            for iteration in range(100):

                # Run one sample at a time
                for graph in graph_samples:

                    loss = self.sess.run([self.train_loss, self.train_step], feed_dict=graph)
                    print("Loss:", loss)



    def sample_train(self):

        # Sample graph
        graph_sample = {
            self.placeholders['node_token_ids']: np.zeros((3, 64)),
            self.placeholders['num_incoming_edges_per_type']: np.zeros((3, self.params['n_edge_types']), dtype=np.float32),
            self.placeholders['slot_ids']: np.zeros((4,), dtype=np.int32),
            self.placeholders['decoder_targets']: np.zeros((32, 1)),
            self.placeholders['decoder_targets_length']: np.ones((1)),
            self.placeholders['target_mask']: np.ones((32, 1))
        }

        for i in range(self.params['n_edge_types']):
            graph_sample[self.placeholders['adjacency_lists'][i]] = np.zeros((10, 2))


        self.train([graph_sample])
        print("Trained...")






def get_gnn_params():

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



def compute_adjacency_lists(graph):

    sorted_edges = sorted(graph.edge, key=lambda x: (int(x.type), int(x.sourceId), int(x.destinationId)), reverse=False)
    grouped = [list(g) for k, g in groupby(sorted_edges, lambda x: x.type)]
    adjacency_tensors = [[[x.sourceId, x.destinationId] for x in group] for group in grouped]
    adjacency_tensors = [np.asarray(data, np.int64) for data in adjacency_tensors]

    return adjacency_tensors


def compute_edges_per_type(graph, incoming=True):

    # n_nodes = ... #TODO: ensure ids are consecutive

    # TODO: assumes node ids range from 0 to n_nodes. Ensure this assumption actually holds
    n_nodes, n_edge_types = 10, 5
    edges_matrix = np.zeros((n_nodes, n_edge_types), np.int64)

    if incoming:
        for e in graph.edge: edges_matrix[e.destinationId, e.type] += 1
    else:
        for e in graph.edge: edges_matrix[e.sourceId, e.type] += 1

    return edges_matrix



def main(path):

  with open(path, "rb") as f:

    g = Graph()
    g.ParseFromString(f.read())

    m = model()
    m.sample_train()




main("/Users/AdminDK/Dropbox/Part III Modules/R252 Machine Learning "
     "for Programming/Practicals/features-javac-master/Test.java.proto")





'''
Nodes:
FAKE_AST, SYMBOL, SYMBOL_TYP, SYMBOL_VAR, SYMBOL_MTH, COMMENT_LINE, COMMENT_BLOCK, COMMENT_JAVADOC ?

Used Node Types: TOKEN, AST_ELEMENT, IDENTIFIER_TOKEN


Edges:
ASSOCIATED_TOKEN, NONE, COMMENT

Used Edge Types: NEXT_TOKEN, AST_CHILD, LAST_WRITE, LAST_USE, COMPUTED_FROM, RETURNS_TO, FORMAL_ARG_NAME, GUARDED_BY,
GUARDED_BY_NEGATION, LAST_LEXICAL_USE, 
'''






