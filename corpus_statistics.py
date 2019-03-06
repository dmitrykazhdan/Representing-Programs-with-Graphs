import os
import yaml
from graph_pb2 import Graph
from dpu_utils.codeutils import split_identifier_into_parts
from graph_pb2 import FeatureNode
import graph_preprocessing


def compute_corpus_stats():

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    corpus_path = cfg['corpus_path']

    max_node_len, max_var_len, max_var_usage = 0, 0, 0

    for dirpath, dirs, files in os.walk(corpus_path):
        for filename in files:
            if filename.endswith('proto'):

                fname = os.path.join(dirpath, filename)

                with open(fname, "rb") as f:

                    g = Graph()
                    g.ParseFromString(f.read())

                    sym_var_node_usages = {}
                    identifier_node_ids = []

                    for node in g.node:

                        if node.type not in graph_preprocessing.get_used_nodes_type() \
                                and node.type != FeatureNode.SYMBOL_VAR: continue

                        node_len = len(split_identifier_into_parts(node.contents))

                        if node_len > max_node_len: max_node_len = node_len

                        if node.type == FeatureNode.SYMBOL_VAR:

                            sym_var_node_usages[node.id] = 0

                            if node_len > max_var_len: max_var_len = node_len


                        elif node.type == FeatureNode.IDENTIFIER_TOKEN:
                            identifier_node_ids.append(node.id)


                    for edge in g.edge:

                        if edge.sourceId in sym_var_node_usages and edge.destinationId in identifier_node_ids:
                            sym_var_node_usages[edge.sourceId] += 1


                    if len(sym_var_node_usages.values()) > 0:
                        var_usage = max(sym_var_node_usages.values())
                    else:
                        var_usage = 0

                    if var_usage > max_var_usage: max_var_usage = var_usage


    print("Longest node length: ", max_node_len)
    print("Longest variable length: ", max_var_len)
    print("Largest variable usage: ", max_var_usage)


compute_corpus_stats()




