import os
from graph_pb2 import Graph
from dpu_utils.codeutils import split_identifier_into_parts
from graph_pb2 import FeatureNode
import graph_processing
from collections import defaultdict



def compute_successors_and_predecessors(graph):

    successor_table = defaultdict(set)
    predecessor_table = defaultdict(set)

    for edge in graph.edge:
        successor_table[edge.sourceId].add(edge.destinationId)
        predecessor_table[edge.destinationId].add(edge.sourceId)

    return successor_table, predecessor_table


def compute_node_table(graph):

    id_dict = {}

    for node in graph.node:
        id_dict[node.id] = node

    return id_dict





def compute_f1_score(prediction, test_label):

    pred_copy = prediction.copy()
    tp = 0

    for subtoken in set(test_label):
        if subtoken in pred_copy:
            tp += 1
            pred_copy.remove(subtoken)


    if len(prediction) > 0:
        pr = tp / len(prediction)
    else:
        pr = 0

    if len(test_label) > 0:
        rec = tp / len(test_label)
    else:
        rec = 0


    if (pr + rec) > 0:
        f1 = 2 * pr * rec / (pr + rec)
    else:
        f1 = 0

    return f1





# Compute corpus metrics in order make a more informed model hyperparameter selection
def compute_corpus_stats(corpus_path):

    max_node_len, max_var_len, max_var_usage = 0, 0, 0

    for dirpath, dirs, files in os.walk(corpus_path):
        for filename in files:
            if filename.endswith('proto'):

                fname = os.path.join(dirpath, filename)

                with open(fname, "rb") as f:

                    g = Graph()
                    g.ParseFromString(f.read())

                    var_node_usages = {}
                    identifier_node_ids = []

                    for node in g.node:

                        if node.type not in graph_processing.get_used_nodes_type() \
                                and node.type != FeatureNode.SYMBOL_VAR:
                            continue

                        node_len = len(split_identifier_into_parts(node.contents))

                        if node_len > max_node_len:
                            max_node_len = node_len

                        if node.type == FeatureNode.SYMBOL_VAR:

                            var_node_usages[node.id] = 0

                            if node_len > max_var_len:
                                max_var_len = node_len


                        elif node.type == FeatureNode.IDENTIFIER_TOKEN:
                            identifier_node_ids.append(node.id)


                    for edge in g.edge:

                        if edge.sourceId in var_node_usages and edge.destinationId in identifier_node_ids:
                            var_node_usages[edge.sourceId] += 1


                    if len(var_node_usages.values()) > 0:
                        var_usage = max(var_node_usages.values())
                    else:
                        var_usage = 0

                    if var_usage > max_var_usage: max_var_usage = var_usage


    print("Longest node length: ", max_node_len)
    print("Longest variable length: ", max_var_len)
    print("Largest variable usage: ", max_var_usage)
