import tensorflow as tf
from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from dpu_utils.tfmodels import SparseGGNN
from typing import List, Optional, Dict, Any
from itertools import groupby
import numpy as np




class model():

    def __init__(self):

        self.params = get_gnn_params()
        self.voc_size = 100
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.embedding_size = 64
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            self.sess.run(init_op)


    def make_model(self):

        h_dim = self.params['hidden_size']

        # Padded graph node sub-token sequences
        self.placeholders['node_token_ids'] = tf.placeholder(tf.int32, [None, h_dim],
                                                                          name='node_features')

        # Graph adjacency lists
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.params['n_edge_types'])]

        # Graph of incoming edges per type
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.params['n_edge_types']],
                                                                          name='num_incoming_edges_per_type')

        # Node identifiers of all graph nodes of the target variable
        self.placeholders['slot_ids'] = tf.placeholder(tf.int32, [None], name='slot_tokens')

        # Actual variable name, as a padded sequence of tokens
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='train_label')
        self.decoder_targets_length = tf.placeholder(shape = (None,), dtype = tf.int32, name = 'decoder_targets_length')

        # Compute the embedding of input node sub-tokens
        self.embedding_encoder = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size]))
        self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_encoder, self.placeholders['node_token_ids'])


        # Average the sub-token embeddings for every node
        self.placeholders['averaged_initial_representation'] = tf.reduce_mean(self.embedding_inputs,
                                                                              axis=1)

        # Run graph through GGNN
        self.gnn_model = SparseGGNN(self.params)
        self.placeholders['gnn_reps'] = self.gnn_model.sparse_gnn_layer(1.0,
                                                                        self.placeholders['averaged_initial_representation'],
                                                                        self.placeholders['adjacency_lists'],
                                                                        self.placeholders['num_incoming_edges_per_type'],
                                                                        self.placeholders['num_incoming_edges_per_type'],
                                                                        {})


        # Compute average of <SLOT> usage representations
        self.placeholders['avg_reps'] = tf.expand_dims(tf.reduce_mean(tf.gather(self.placeholders['gnn_reps'],
                                                                                self.placeholders['slot_ids']), axis=0), 0)

        self.target_mask = tf.placeholder(tf.float32, [None, None], name='train_label')

        # Obtain output sequence by passing through a single GRU layer
        self.decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32,name='decoder_targets_length')
        self.embedding_decoder = tf.Variable(tf.random_uniform([self.voc_size, self.embedding_size]))
        self.decoder_embedding_inputs = tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_targets)

        decoder_cell = tf.nn.rnn_cell.GRUCell(self.params['hidden_size'])
        initial_state = self.placeholders['avg_reps']
        projection_layer = tf.layers.Dense(self.voc_size, use_bias=False)

        helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embedding_inputs, self.decoder_targets_length)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state=initial_state, output_layer=projection_layer)

        self.decoder_outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        self.decoder_logits_train = self.decoder_outputs_train.rnn_output

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets, logits=self.decoder_logits_train)
        loss = tf.reduce_sum(self.target_mask * crossent)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, 5.0)

        # Optimization
        optimizer = tf.train.AdamOptimizer(0.01)
        train_op = optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=global_step)


        print("Model built successfully...")


        # TODO: Add inference code to GRU layer for predictions




    # TODO: fill in method
    def make_train_step(self):
        return None



    def train(self, graph_samples):

        with self.graph.as_default():

            n_samples = len(graph_samples)

            for iteration in range(20):

                # Run one sample at a time
                for sample in range(n_samples):

                    graph = graph_samples[sample]
                    loss = self.sess.run([self.ops['loss'], self.train_step], feed_dict=graph)
                    print("Loss=", loss)



    def sample_train(self):

        # Sample graph
        graph_sample = {
            self.placeholders['var_name']: np.ones((1, 1, 20), dtype=np.float32),
            self.placeholders['node_token_ids']: np.zeros((3, 64)),
            self.placeholders['num_incoming_edges_per_type']: np.zeros((3, self.params['n_edge_types']), dtype=np.float32),
            self.placeholders['slot_ids']: np.zeros((4,), dtype=np.int32)
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
    gnn_params["add_backwards_edges"] = False
    gnn_params["message_aggregation_type"] = "sum"
    gnn_params["layer_timesteps"] = [8]
    gnn_params["use_propagation_attention"] = False
    gnn_params["use_edge_bias"] = False
    gnn_params["graph_rnn_activation"] = "relu"
    gnn_params["graph_rnn_cell"] = "gru"
    gnn_params["residual_connections"] = {}  #
    gnn_params["use_edge_msg_avg_aggregation"] = False

    return gnn_params


# TODO: retrieve adjacency list, given graph
def compute_adjacency_lists(graph) -> List[tf.Tensor]:

    sorted_edges = sorted(graph.edge, key=lambda x: (int(x.type), int(x.sourceId), int(x.destinationId)), reverse=False)
    grouped = [list(g) for k, g in groupby(sorted_edges, lambda x: x.type)]
    adjacency_tensors = [[[x.sourceId, x.destinationId] for x in group] for group in grouped]
    adjacency_tensors = [tf.convert_to_tensor(np.asarray(data, np.int64), np.int64) for data in adjacency_tensors]

    return adjacency_tensors



# TODO: compute incoming/outgoing edges per type
def compute_edges_per_type(graph, incoming=True) -> tf.Tensor:

    # n_nodes = ... #TODO: ensure ids are consecutive

    # TODO: assumes node ids range from 0 to n_nodes. Ensure this assumption actually holds
    n_nodes, n_edge_types = 10, 5
    edges_matrix = np.zeros((n_nodes, n_edge_types), np.int64)

    if incoming:
        for e in graph.edge: edges_matrix[e.destinationId, e.type] += 1
    else:
        for e in graph.edge: edges_matrix[e.sourceId, e.type] += 1

    edges_matrix = tf.convert_to_tensor(edges_matrix, np.int64)

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






