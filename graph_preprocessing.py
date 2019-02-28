from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from collections import defaultdict
import numpy as np
from dpu_utils.codeutils import split_identifier_into_parts



def get_used_edges_type():

    used_edge_types = [FeatureEdge.NEXT_TOKEN, FeatureEdge.AST_CHILD, FeatureEdge.LAST_WRITE,
                       FeatureEdge.LAST_USE, FeatureEdge.COMPUTED_FROM, FeatureEdge.RETURNS_TO,
                       FeatureEdge.FORMAL_ARG_NAME, FeatureEdge.GUARDED_BY, FeatureEdge.GUARDED_BY_NEGATION,
                       FeatureEdge.LAST_LEXICAL_USE]

    return used_edge_types



def get_used_nodes_type():

    used_node_types = [FeatureNode.TOKEN, FeatureNode.AST_ELEMENT, FeatureNode.IDENTIFIER_TOKEN]

    return used_node_types





def compute_sample_data(graph, seq_length, pad_token, vocabulary):

    used_node_types = get_used_nodes_type()
    used_edge_types = get_used_edges_type()

    node_representations = []
    var_nodes_map = defaultdict(list)
    id_to_index_map = {}
    identifier_token_node_ids = []
    symbol_var_node_ids = []
    ind = 0

    for node in graph.node:
        if node.type in used_node_types:
            node_representation = vocabulary.get_id_or_unk_multiple(split_identifier_into_parts(node.contents), seq_length, pad_token)
            node_representations.append(node_representation)
            id_to_index_map[node.id] = ind
            ind += 1

        if node.type == FeatureNode.IDENTIFIER_TOKEN:
            identifier_token_node_ids.append(node.id)
        elif node.type == FeatureNode.SYMBOL_VAR:
            symbol_var_node_ids.append(node.id)


    n_nodes = len(node_representations)
    n_types = len(used_edge_types)
    node_representations = np.array(node_representations)
    num_incoming_edges_per_type = np.zeros((n_nodes, n_types))
    num_outgoing_edges_per_type = np.zeros((n_nodes, n_types))
    adj_lists = defaultdict(list)

    for edge in graph.edge:
        if edge.type in used_edge_types \
                and edge.sourceId in id_to_index_map \
                and edge.destinationId in id_to_index_map:

            type_id = used_edge_types.index(edge.type)
            adj_lists[type_id].append([id_to_index_map[edge.sourceId], id_to_index_map[edge.destinationId]])
            num_incoming_edges_per_type[id_to_index_map[edge.destinationId], type_id] += 1
            num_outgoing_edges_per_type[id_to_index_map[edge.sourceId], type_id] += 1

        if edge.sourceId in symbol_var_node_ids and edge.destinationId in identifier_token_node_ids:
            var_nodes_map[edge.sourceId].append(id_to_index_map[edge.destinationId])



    final_adj_lists = {edge_type: np.array(sorted(adj_list), dtype=np.int32)
                       for edge_type, adj_list in adj_lists.items()}

    # Add empty entries for types with no adjacency lists
    for i in range(len(used_edge_types)):
        if i not in final_adj_lists:
            final_adj_lists[i] = np.zeros((0, 2), dtype=np.int32)


    return var_nodes_map, node_representations, final_adj_lists, \
           num_incoming_edges_per_type, num_outgoing_edges_per_type








