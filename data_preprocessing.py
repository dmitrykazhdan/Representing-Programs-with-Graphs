from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np


class SampleMetaInformation():

    def __init__(self, sample_fname, node_id):

        self.fname = sample_fname
        self.node_id = node_id
        self.predicted_correctly = None
        self.type = "empty_type"
        self.usages = None
        self.usage_rep = None
        self.true_label = None
        self.predicted_label = None
        self.seen_in_training = None


    def compute_var_type(self):

        if self.type != "empty_type": return self.type


        with open(self.fname, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            var_type = get_var_type(g, self.node_id)

        self.type = var_type

        return var_type



    def compute_var_usages(self):

        return None

        if self.usages != None: return self.usages

        with open(self.fname, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            n_usages = get_var_usages(g, self.node_id)

        self.usages = n_usages

        return n_usages




class CorpusMetaInformation():

    def __init__(self, _sample_meta_inf):
        self.sample_meta_inf = _sample_meta_inf


    def add_sample_inf(self, sample_inf):
        self.sample_meta_inf.append(sample_inf)


    def compute_usage_clusters(self):

        usage_arr = [sample_inf.usage_rep for sample_inf in self.sample_meta_inf]
        usage_arr = np.vstack(usage_arr)

        kmeans = KMeans(n_clusters=10).fit(usage_arr)

        labels_with_names = [[kmeans.labels_[i], self.sample_meta_inf[i].true_label, self.predicted_label]
                             for i in range(len(self.sample_meta_inf)) if self.sample_meta_inf[i].predicted_correctly == True]

        labels_with_names = sorted(labels_with_names, key=lambda x: x[0])

        print(labels_with_names)






    def process_sample_inf(self):

        incorr_usage_classes, corr_usage_classes = defaultdict(int), defaultdict(int)
        incorr_type_classes, corr_type_classes = defaultdict(int), defaultdict(int)

        for sample_inf in self.sample_meta_inf:

            if sample_inf.seen_in_training:
                #print("filename: ", sample_inf.fname)
                sample_inf.compute_var_usages()
                sample_inf.compute_var_type()

                if sample_inf.predicted_correctly:
                    corr_usage_classes[sample_inf.usages] += 1
                    corr_type_classes[sample_inf.type] += 1
                else:
                    incorr_usage_classes[sample_inf.usages] += 1
                    incorr_type_classes[sample_inf.type] += 1



            #print("Usage: ", sample_inf.usages)
            #print("Type: ", sample_inf.type)


        # for i in range(-5, 128, 1):
        #
        #     if incorr_usage_classes[i] != 0 or  corr_usage_classes[i] != 0:
        #         print(str(i) + " usages: ", incorr_usage_classes[i], " (incorrect)      ", corr_usage_classes[i], " (correct)")
        #         print("")



        for key in list(set(incorr_type_classes.keys()).union(corr_type_classes.keys())):

            if incorr_type_classes[key] != 0 or corr_type_classes[key] != 0:
                print(str(key), incorr_type_classes[key], " (incorrect)      ", corr_type_classes[key], " (correct)")
                print("")









# Pre-compute successors and predecessors for a given graph
def compute_successors_and_predecessors(graph):

    successor_table = defaultdict(set)
    predecessor_table = defaultdict(set)

    for edge in graph.edge:
        successor_table[edge.sourceId].add(edge.destinationId)
        predecessor_table[edge.destinationId].add(edge.sourceId)

    return successor_table, predecessor_table


def compute_id_dict(graph):

    id_dict = {}

    for node in graph.node:
        id_dict[node.id] = node

    return id_dict



def get_var_type(graph, sym_var_node_id):

    id_dict = compute_id_dict(graph)
    successors, predecessors = compute_successors_and_predecessors(graph)

    id_token_nodes = [n_id for n_id in successors[sym_var_node_id] if id_dict[n_id].type == FeatureNode.IDENTIFIER_TOKEN]

    ast_parent = -1

    for id_token_node in id_token_nodes:
        for parent_id in predecessors[id_token_node]:

            if id_dict[parent_id].type == FeatureNode.AST_ELEMENT and id_dict[parent_id].contents == "VARIABLE":
                ast_parent = parent_id
                break

        if ast_parent != -1: break


    if ast_parent == -1:
        return "empty_type"


    fake_ast_type = [n for n in successors[ast_parent]
                     if id_dict[n].type == FeatureNode.FAKE_AST and id_dict[n].contents == "TYPE"]

    if len(fake_ast_type) == 0:
        return "empty_type"

    else:
        fake_ast_type = fake_ast_type[0]


    fake_ast_type_succ = list(successors[fake_ast_type])

    if len(fake_ast_type_succ) == 0:
        return "empty_type"

    else:
        fake_ast_type_succ = fake_ast_type_succ[0]


    type = [id_dict[n].contents for n in successors[fake_ast_type_succ] if id_dict[n].type == FeatureNode.TYPE]

    if len(type) == 0:
        return "empty_type"

    else:
        type = type[0]

    print("type: ", type)

    return type




# Find all Identifier Token successors of a SYM_VAR node
def get_var_usages(graph, var_id):

    id_dict = compute_id_dict(graph)

    usages = 0

    for edge in graph.edge:
        if edge.sourceId == var_id:
            dest_id = edge.destinationId
            child_node = id_dict[dest_id]

            if child_node.type == FeatureNode.IDENTIFIER_TOKEN:
                usages+=1


    return usages





