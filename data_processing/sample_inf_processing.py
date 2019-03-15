from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from collections import defaultdict
from utils.utils import compute_successors_and_predecessors, compute_node_table


class SampleMetaInformation():

    def __init__(self, sample_fname, node_id):

        self.fname = sample_fname
        self.node_id = node_id
        self.predicted_correctly = None
        self.empty_type = "undefined"
        self.type = self.empty_type
        self.num_usages = None
        self.usage_rep = None
        self.true_label = None
        self.predicted_label = None
        self.seen_in_training = None


    def compute_var_type(self):

        if self.type != self.empty_type: return self.type


        with open(self.fname, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            var_type = get_var_type(g, self.node_id, self.empty_type)

        self.type = var_type

        return var_type



    def compute_var_usages(self):

        if self.num_usages is not None: return self.num_usages

        with open(self.fname, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            n_usages = get_var_usages(g, self.node_id)

        self.num_usages = n_usages

        return n_usages




class CorpusMetaInformation():

    def __init__(self, _sample_meta_infs):
        self.sample_meta_infs = _sample_meta_infs


    def add_sample_inf(self, sample_inf):
        self.sample_meta_infs.append(sample_inf)


    def process_sample_inf(self):

        incorr_usage_classes, corr_usage_classes = defaultdict(int), defaultdict(int)
        incorr_type_classes, corr_type_classes = defaultdict(int), defaultdict(int)


        # Compute and print usage and type information from entire corpus
        for sample_inf in self.sample_meta_infs:

            if sample_inf.seen_in_training:

                sample_inf.compute_var_usages()
                sample_inf.compute_var_type()

                if sample_inf.predicted_correctly:
                    corr_usage_classes[sample_inf.num_usages] += 1
                    corr_type_classes[sample_inf.type] += 1
                else:
                    incorr_usage_classes[sample_inf.num_usages] += 1
                    incorr_type_classes[sample_inf.type] += 1



        # Print the computed information:
        all_usage_keys = list(set(incorr_usage_classes.keys()).union(corr_usage_classes.keys()))

        for usage_key in all_usage_keys:
            print(str(usage_key) + " usages: ", incorr_usage_classes[usage_key], " (incorrect) ", corr_usage_classes[usage_key], " (correct)")
            print("")



        all_type_keys = list(set(incorr_type_classes.keys()).union(corr_type_classes.keys()))

        for type_key in all_type_keys:
            print(str(type_key), incorr_type_classes[type_key], " (incorrect) ", corr_type_classes[type_key], " (correct)")
            print("")




def get_var_type(graph, sym_var_node_id, empty_type):

    node_table = compute_node_table(graph)
    successors, predecessors = compute_successors_and_predecessors(graph)

    id_token_nodes = [n_id for n_id in successors[sym_var_node_id] if node_table[n_id].type == FeatureNode.IDENTIFIER_TOKEN]

    ast_parent = -1

    for id_token_node in id_token_nodes:
        for parent_id in predecessors[id_token_node]:

            if node_table[parent_id].type == FeatureNode.AST_ELEMENT and node_table[parent_id].contents == "VARIABLE":
                ast_parent = parent_id
                break

        if ast_parent != -1: break


    if ast_parent == -1: return empty_type


    fake_ast_type_nodes = [n for n in successors[ast_parent]
                     if node_table[n].type == FeatureNode.FAKE_AST and node_table[n].contents == "TYPE"]

    if len(fake_ast_type_nodes) == 0:
        return empty_type

    else:
        fake_ast_type_node = fake_ast_type_nodes[0]


    fake_ast_type_successors = list(successors[fake_ast_type_node])

    if len(fake_ast_type_successors) == 0:
        return "empty_type"

    else:
        fake_ast_type_successor = fake_ast_type_successors[0]


    type_contents = [node_table[n].contents for n in successors[fake_ast_type_successor] if node_table[n].type == FeatureNode.TYPE]

    if len(type_contents) == 0:
        return empty_type

    else:
        type_content = type_contents[0]

    return type_content




# Find all Identifier Token successors of a SYM_VAR node
def get_var_usages(graph, var_id):

    node_table = compute_node_table(graph)

    usages = 0

    for edge in graph.edge:
        if edge.sourceId == var_id:
            dest_id = edge.destinationId
            child_node = node_table[dest_id]

            if child_node.type == FeatureNode.IDENTIFIER_TOKEN:
                usages += 1

    return usages





