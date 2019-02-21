from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from collections import defaultdict
import numpy as np
from dpu_utils.codeutils import split_identifier_into_parts

# Filter out nodes/edges from graph
def filter_graph(graph):

    used_node_types = [FeatureNode.TOKEN, FeatureNode.AST_ELEMENT, FeatureNode.IDENTIFIER_TOKEN,
                       FeatureNode.FAKE_AST, FeatureNode.COMMENT_LINE, FeatureNode.COMMENT_BLOCK]

    used_edge_types = [FeatureEdge.NEXT_TOKEN, FeatureEdge.AST_CHILD, FeatureEdge.LAST_WRITE,
                       FeatureEdge.LAST_USE, FeatureEdge.COMPUTED_FROM, FeatureEdge.RETURNS_TO,
                       FeatureEdge.FORMAL_ARG_NAME, FeatureEdge.GUARDED_BY, FeatureEdge.GUARDED_BY_NEGATION,
                       FeatureEdge.LAST_LEXICAL_USE]


    filtered_nodes = [node for node in graph.node if node.type in used_node_types]
    filtered_node_ids = [node.id for node in filtered_nodes]

    filtered_edges = [edge for edge in graph.edge if edge.type in used_edge_types
                      and edge.sourceId in filtered_node_ids
                      and edge.destinationId in filtered_node_ids]

    return filtered_nodes, filtered_edges




# Aquire map from node id in the graph to the node index in the node representation matrix
def get_node_id_to_index_map(nodes):

    id_to_index_map = {}

    ind = 0
    for node in nodes:
        id_to_index_map[node.id] = ind
        ind += 1

    return id_to_index_map



# Obtain map from symbol_var node id to all corresponding variable identifier tokens
def get_var_nodes_map(graph, id_to_index_map):

    var_nodes_map = defaultdict(list)

    # Extract node ids of all identifier tokens
    identifier_token_node_ids = [node.id for node in graph.node if node.type == FeatureNode.IDENTIFIER_TOKEN]

    # Extract node ids of all symbol variable nodes
    symbol_var_node_ids = [node.id for node in graph.node if node.type == FeatureNode.SYMBOL_VAR]


    # Assume all identifier nodes are direct descendants of a symbol variable node
    for edge in graph.edge:
        if edge.sourceId in symbol_var_node_ids and edge.destinationId in identifier_token_node_ids:
            var_nodes_map[edge.sourceId].append(id_to_index_map[edge.destinationId])

    return var_nodes_map




def compute_initial_node_representation(nodes, seq_length, pad_token, vocabulary):

    node_representations = np.array([vocabulary.get_id_or_unk_multiple(split_identifier_into_parts(node.contents),
                                                                            seq_length, pad_token)
                                                                        for node in nodes])

    return node_representations




def compute_adjacency_lists(edges, id_to_index_map):

    adj_lists = defaultdict(list)

    used_edge_types = [FeatureEdge.NEXT_TOKEN, FeatureEdge.AST_CHILD, FeatureEdge.LAST_WRITE,
                       FeatureEdge.LAST_USE, FeatureEdge.COMPUTED_FROM, FeatureEdge.RETURNS_TO,
                       FeatureEdge.FORMAL_ARG_NAME, FeatureEdge.GUARDED_BY, FeatureEdge.GUARDED_BY_NEGATION,
                       FeatureEdge.LAST_LEXICAL_USE]


    for edge in edges:
        type_id = used_edge_types.index(edge.type)
        adj_lists[type_id].append([id_to_index_map[edge.sourceId], id_to_index_map[edge.destinationId]])


    # Temporary solution: add dummy edge to have at least one edge per type
    dummy_edge = [id_to_index_map[edges[0].sourceId], id_to_index_map[edges[0].destinationId]]




    for i in range(10):
        if i not in adj_lists:
            adj_lists[i].append(dummy_edge)


    print("Types: ", len(adj_lists))

    final_adj_lists = {edge_type: np.array(sorted(adj_list), dtype=np.int32)
                       for edge_type, adj_list in adj_lists.items()}


    return final_adj_lists





def compute_edges_per_type(n_nodes, adj_lists):

    n_types = len(adj_lists)

    num_incoming_edges_per_type = np.zeros((n_nodes, n_types))
    num_outgoing_edges_per_type = np.zeros((n_nodes, n_types))

    i = 0

    for edge_type in adj_lists:

        adj_list = adj_lists[edge_type]

        for edge in adj_list:
            num_incoming_edges_per_type[edge[1], i] += 1
            num_outgoing_edges_per_type[edge[0], i] += 1

        i += 1


    return num_incoming_edges_per_type, num_outgoing_edges_per_type