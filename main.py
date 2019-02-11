import tensorflow as tf
from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from dpu_utils.tfmodels import SparseGGNN
from typing import List, Optional, Dict, Any
from itertools import groupby
import numpy as np




class model():

    def __init__(self):

        self.params= get_gnn_params()
        self.gnn_model = SparseGGNN(self.params)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.placeholders = {}
        self.weights = {}
        self.ops = {}
        self.make_model()




    def make_model(self):

        h_dim = self.params['hidden_size']
        self.num_edge_types=10

        self.placeholders['var_name'] = tf.placeholder(tf.float32, [1, 1, 20], name='train_label')

        self.placeholders['node_token_ids'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='node_features')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')

        self.placeholders['slot_ids'] = tf.placeholder(tf.int32, [None, 1], name='slot_tokens')


        # self.placeholders['initial_node_representation'] = tf.keras.layers.Embedding(self.placeholders['node_token_ids'].shape, )


        self.placeholders['gnn_reps'] = self.gnn_model.sparse_gnn_layer(1.0,
                                                                        self.placeholders['initial_node_representation'],
                                                                        self.placeholders['adjacency_lists'],
                                                                        self.placeholders['num_incoming_edges_per_type'],
                                                                        self.placeholders['num_incoming_edges_per_type'],
                                                                        {})


        self.placeholders['avg_reps'] = tf.expand_dims(tf.reduce_mean(tf.gather(self.placeholders['gnn_reps'],
                                                                                self.placeholders['slot_ids']), axis=0), 0)

        self.gru_layer = tf.keras.layers.GRU(20, activation='relu', return_sequences=True)

        self.placeholders['output_seq'] = self.gru_layer(self.placeholders['avg_reps'])



        self.ops['loss'] = tf.reduce_mean(tf.squared_difference(self.placeholders['var_name'], self.placeholders['output_seq']))
        my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
        self.train_step = my_opt.minimize(self.ops['loss'])



    def train(self, graph_samples):

        n_samples = 10

        for iteration in range(100):
            for sample in range(n_samples):

                graph = graph_samples[sample]
                loss = self.sess.run([self.ops['loss'], self.train_step], feed_dict=graph)
                print("Loss=", loss)




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

    # dropout_keep_rate = 1.0
    # node_embeddings=compute_initial_node_embeddings(g)
    # adjacency_lists=compute_adjacency_lists(g)
    # num_incoming_edges_per_type=compute_edges_per_type(g)
    # num_outgoing_edges_per_type=compute_edges_per_type(g, False)
    # edge_features=compute_edge_features(g)


    m = model()




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






