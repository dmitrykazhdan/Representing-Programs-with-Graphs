import tensorflow as tf
from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from dpu_utils.tfmodels import SparseGGNN
from typing import List, Optional, Dict, Any
from itertools import groupby
import numpy as np


# TODO: fixup typo in API
# TODO: check what 'number' means in feature extractor


# TODO: set correct parameter values
def get_gnn_params():

    gnn_params = {}
    gnn_params["n_edge_types"] = 10
    gnn_params["hidden_size"] = 64
    gnn_params["edge_features_size"] = {}  # Dict from edge type to feature size
    gnn_params["add_backwards_edges"] = True
    gnn_params["message_aggregation_type"] = "sum"
    gnn_params["layer_timesteps"] = [8]
    gnn_params["use_propagation_attention"] = True
    gnn_params["use_edge_bias"] = True
    gnn_params["graph_rnn_activation"] = "relu"
    gnn_params["graph_rnn_cell"] = "gru"
    gnn_params["residual_connections"] = None  #
    gnn_params["use_edge_msg_avg_aggregation"] = True

    return gnn_params



'''
Nodes:
FAKE_AST, SYMBOL, SYMBOL_TYP, SYMBOL_VAR, SYMBOL_MTH, COMMENT_LINE, COMMENT_BLOCK, COMMENT_JAVADOC ?

Used Node Types: TOKEN, AST_ELEMENT, IDENTIFIER_TOKEN


Edges:
ASSOCIATED_TOKEN, NONE, COMMENT

Used Edge Types: NEXT_TOKEN, AST_CHILD, LAST_WRITE, LAST_USE, COMPUTED_FROM, RETURNS_TO, FORMAL_ARG_NAME, GUARDED_BY,
GUARDED_BY_NEGATION, LAST_LEXICAL_USE, 
'''




# TODO: retrieve adjacency list, given graph
def compute_adjacency_lists(graph) -> List[tf.Tensor]:

    sorted_edges = sorted(graph.edge, key=lambda x: (int(x.type), int(x.sourceId), int(x.destinationId)), reverse=False)
    grouped = [list(g) for k, g in groupby(sorted_edges, lambda x: x.type)]
    adjacency_tensors = [[[x.sourceId, x.destinationId] for x in group] for group in grouped]
    adjacency_tensors = [tf.convert_to_tensor(np.asarray(data, np.int64), np.int64) for data in adjacency_tensors]

    return adjacency_tensors


# TODO: compute initial node embeddings
def compute_initial_node_embeddings(graph) -> tf.Tensor:
    return None


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



# TODO: compute edge features
def compute_edge_features(graph) -> Dict[int, tf.Tensor]:
    return None



# TODO: run VarNaming task
def run_varnaming(graph, ggnn_model, var_name):

    # Replace var_name with <SLOT> token in graph
    # Run ggnn on graph
    # Average variable usages
    # Run through graph2seq to get variable name

    return None



def main(path):

  # Create sparse GGNN model
  gnn_params = get_gnn_params()
  ggnn = SparseGGNN(gnn_params)

  # TODO: load processed source code repository
  # TODO: split into training/test set
  # TODO: filter out unneeded nodes/edges if necessary
  # TODO: compute type embedding

  with open(path, "rb") as f:

    g = Graph()
    g.ParseFromString(f.read())

    dropout_keep_rate = 1.0
    node_embeddings=compute_initial_node_embeddings(g)
    adjacency_lists=compute_adjacency_lists(g)
    num_incoming_edges_per_type=compute_edges_per_type(g)
    num_outgoing_edges_per_type=compute_edges_per_type(g, False)
    edge_features=compute_edge_features(g)

    node_usages = ggnn.sparse_gnn_layer(dropout_keep_rate, node_embeddings, adjacency_lists,
                        num_incoming_edges_per_type, num_outgoing_edges_per_type, edge_features)





main("/Users/AdminDK/Dropbox/Part III Modules/R252 Machine Learning "
     "for Programming/Practicals/features-javac-master/Test.java.proto")











