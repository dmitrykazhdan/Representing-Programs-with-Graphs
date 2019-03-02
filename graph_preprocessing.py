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

    used_node_types = [FeatureNode.TOKEN, FeatureNode.AST_ELEMENT, FeatureNode.IDENTIFIER_TOKEN,
                       FeatureNode.COMMENT_LINE, FeatureNode.COMMENT_BLOCK, FeatureNode.COMMENT_JAVADOC,
                       FeatureNode.FAKE_AST]

    return used_node_types



def compute_sub_graphs(graph, timesteps, seq_length, pad_token, vocabulary):


    successor_table = defaultdict(set)
    predecessor_table = defaultdict(set)
    edge_table = defaultdict(list)
    node_table = {}
    sym_var_node_ids = []
    samples = []

    for node in graph.node:

        node_table[node.id] = node

        if node.type == FeatureNode.SYMBOL_VAR:
            sym_var_node_ids.append(node.id)


    for edge in graph.edge:
        successor_table[edge.sourceId].add(edge.destinationId)
        predecessor_table[edge.destinationId].add(edge.sourceId)
        edge_table[edge.sourceId].append(edge)


    for sym_var_node_id in sym_var_node_ids:

        successor_ids = list(successor_table[sym_var_node_id])

        var_identifier_node_ids = [node_id for node_id in successor_ids
                                if node_table[node_id].type == FeatureNode.IDENTIFIER_TOKEN]


        # Ensure variable has at least one usage
        if len(var_identifier_node_ids) == 0: continue


        reachable_node_ids = []
        successor_ids = [node_id for node_id in var_identifier_node_ids]
        predecessor_ids = successor_ids

        for _ in range(timesteps):
            reachable_node_ids += successor_ids
            reachable_node_ids += predecessor_ids
            successor_ids = list(set([elem for n_id in successor_ids for elem in list(successor_table[n_id])]))
            predecessor_ids = list(set([elem for n_id in predecessor_ids for elem in list(predecessor_table[n_id])]))

        reachable_node_ids += successor_ids
        reachable_node_ids += predecessor_ids
        reachable_node_ids = list(set(reachable_node_ids))


        sub_nodes = [node_table[node_id] for node_id in reachable_node_ids]

        sub_edges =  [edge for node in sub_nodes for edge in edge_table[node.id]
                      if edge.sourceId in reachable_node_ids and edge.destinationId in reachable_node_ids]

        sub_graph = (sub_nodes, sub_edges)

        sample_data = compute_sample_data(sub_graph, var_identifier_node_ids, seq_length, pad_token, vocabulary)
        samples.append(sample_data)


    return samples








def compute_sample_data(sub_graph, identifier_token_node_ids, seq_length, pad_token, vocabulary):

    used_node_types = get_used_nodes_type()
    used_edge_types = get_used_edges_type()

    node_representations = []
    id_to_index_map = {}
    ind = 0

    (sub_nodes, sub_edges) = sub_graph

    for node in sub_nodes:
        if node.type in used_node_types:
            node_representation = vocabulary.get_id_or_unk_multiple(split_identifier_into_parts(node.contents), seq_length, pad_token)
            node_representations.append(node_representation)
            id_to_index_map[node.id] = ind
            ind += 1

    n_nodes = len(node_representations)
    n_types = len(used_edge_types)
    node_representations = np.array(node_representations)
    num_incoming_edges_per_type = np.zeros((n_nodes, n_types))
    num_outgoing_edges_per_type = np.zeros((n_nodes, n_types))
    adj_lists = defaultdict(list)

    for edge in sub_edges:
        if edge.type in used_edge_types \
                and edge.sourceId in id_to_index_map \
                and edge.destinationId in id_to_index_map:

            type_id = used_edge_types.index(edge.type)
            adj_lists[type_id].append([id_to_index_map[edge.sourceId], id_to_index_map[edge.destinationId]])
            num_incoming_edges_per_type[id_to_index_map[edge.destinationId], type_id] += 1
            num_outgoing_edges_per_type[id_to_index_map[edge.sourceId], type_id] += 1

    final_adj_lists = {edge_type: np.array(sorted(adj_list), dtype=np.int32)
                       for edge_type, adj_list in adj_lists.items()}

    # Add empty entries for types with no adjacency lists
    for i in range(len(used_edge_types)):
        if i not in final_adj_lists:
            final_adj_lists[i] = np.zeros((0, 2), dtype=np.int32)


    var_identifier_nodes = [id_to_index_map[node_id] for node_id in identifier_token_node_ids]

    return (var_identifier_nodes, node_representations, final_adj_lists, \
           num_incoming_edges_per_type, num_outgoing_edges_per_type)








