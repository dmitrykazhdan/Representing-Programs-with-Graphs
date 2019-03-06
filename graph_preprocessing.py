from graph_pb2 import FeatureNode, FeatureEdge
from collections import defaultdict
import numpy as np
from dpu_utils.codeutils import split_identifier_into_parts



def get_used_edges_type():

    used_edge_types = [FeatureEdge.NEXT_TOKEN, FeatureEdge.AST_CHILD, FeatureEdge.LAST_WRITE,
                       FeatureEdge.LAST_USE, FeatureEdge.COMPUTED_FROM, FeatureEdge.RETURNS_TO,
                       FeatureEdge.FORMAL_ARG_NAME, FeatureEdge.GUARDED_BY, FeatureEdge.GUARDED_BY_NEGATION,
                       FeatureEdge.LAST_LEXICAL_USE,
                       FeatureEdge.ASSIGNABLE_TO, FeatureEdge.ASSOCIATED_TOKEN,
                       FeatureEdge.HAS_TYPE, FeatureEdge.ASSOCIATED_SYMBOL]

    return used_edge_types



def get_used_nodes_type():

    used_node_types = [FeatureNode.TOKEN, FeatureNode.AST_ELEMENT, FeatureNode.IDENTIFIER_TOKEN,
                       FeatureNode.COMMENT_LINE,
                       FeatureNode.FAKE_AST, FeatureNode.SYMBOL_TYP,
                       FeatureNode.TYPE]

    return used_node_types



def compute_sub_graphs(graph, timesteps, var_seq_length, node_seq_length, pad_token, slot_token, vocabulary, method_nodes=False):

    successor_table = defaultdict(set)
    predecessor_table = defaultdict(set)
    edge_table = defaultdict(list)
    node_table = {}
    sym_var_node_ids = []
    samples = []
    non_empty_sym_nodes = []


    if method_nodes:
        parent_usage_node_type = FeatureNode.SYMBOL_MTH
    else:
        parent_usage_node_type = FeatureNode.SYMBOL_VAR


    for node in graph.node:

        node_table[node.id] = node

        if node.type == parent_usage_node_type:
            sym_var_node_ids.append(node.id)




    for edge in graph.edge:
        successor_table[edge.sourceId].add(edge.destinationId)
        predecessor_table[edge.destinationId].add(edge.sourceId)
        edge_table[edge.sourceId].append(edge)



    for sym_var_node_id in sym_var_node_ids:

        successor_ids = successor_table[sym_var_node_id]

        var_identifier_node_ids = [node_id for node_id in successor_ids
                                if node_table[node_id].type == FeatureNode.IDENTIFIER_TOKEN]

        decl_id_nodes = []

        # If doing method processing, need to also check for presence of the method declaration
        if method_nodes:
            ast_elem_successors = [node_id for node_id in successor_ids
                                   if node_table[node_id].type==FeatureNode.AST_ELEMENT and node_table[node_id].contents=='METHOD']

            if len(ast_elem_successors) > 0:

                method_decl_node_id = ast_elem_successors[0]

                decl_id_nodes = [node_id for node_id in successor_table[method_decl_node_id] if node_table[node_id].type == FeatureNode.IDENTIFIER_TOKEN]



        # Ensure variable has at least one usage
        if len(var_identifier_node_ids) == 0 or len(var_identifier_node_ids) > var_seq_length: continue


        reachable_node_ids = []
        successor_ids = var_identifier_node_ids
        predecessor_ids = var_identifier_node_ids

        for _ in range(timesteps):
            reachable_node_ids += successor_ids
            reachable_node_ids += predecessor_ids
            successor_ids = list(set([elem for n_id in successor_ids for elem in successor_table[n_id]]))
            predecessor_ids = list(set([elem for n_id in predecessor_ids for elem in predecessor_table[n_id]]))

        reachable_node_ids += successor_ids
        reachable_node_ids += predecessor_ids
        reachable_node_ids = set(reachable_node_ids)


        sub_nodes = [node_table[node_id] for node_id in reachable_node_ids]

        sub_edges =  [edge for node in sub_nodes for edge in edge_table[node.id]
                      if edge.sourceId in reachable_node_ids and edge.destinationId in reachable_node_ids]

        sub_graph = (sub_nodes, sub_edges)

        sample_data = compute_sample_data(sub_graph, var_identifier_node_ids, node_seq_length, pad_token, slot_token, vocabulary, decl_id_nodes)
        samples.append(sample_data)
        non_empty_sym_nodes.append(sym_var_node_id)

    return samples, non_empty_sym_nodes








def compute_sample_data(sub_graph, identifier_token_node_ids, seq_length, pad_token, slot_token, vocabulary, exception_node_ids = []):

    used_node_types = get_used_nodes_type()
    used_edge_types = get_used_edges_type()

    node_representations = []
    id_to_index_map = {}
    ind = 0

    (sub_nodes, sub_edges) = sub_graph

    for node in sub_nodes:
        if node.type in used_node_types:

            if node.id in exception_node_ids:
                node_representation = [pad_token for _ in range(seq_length)]
                node_representation[0] = slot_token
            else:
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



