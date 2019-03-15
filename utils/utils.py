import os
from graph_pb2 import Graph
from dpu_utils.codeutils import split_identifier_into_parts
from graph_pb2 import FeatureNode
from data_processing.graph_features import get_used_nodes_type
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

                        if node.type not in get_used_nodes_type() \
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



# Used for parsing type information from a type file
def get_type_dists(types_fname):

    content_arr = []

    with open(types_fname, "r") as f:

        for line in f:

            line_contents = str.split(line.strip())

            if len(line_contents) == 5:

                line_contents = [line_contents[0], int(line_contents[1]), int(line_contents[3])]

                if line_contents[1] + line_contents[2] > 100:
                    content_arr.append(line_contents)


        pred_acc = [[inf[0], 100 * inf[2] / (inf[2] + inf[1])] for inf in content_arr]
        pred_acc = sorted(pred_acc, key=lambda  x: x[1], reverse=True)

        names = [inf[0] for inf in pred_acc]
        percentages = [inf[1] for inf in pred_acc]


    return names, percentages




