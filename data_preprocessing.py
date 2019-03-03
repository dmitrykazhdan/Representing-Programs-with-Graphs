from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from collections import defaultdict


# Currently sample meta-information consists of the graph filename and the sym_var node id
class SampleMetaInformation():

    def __init__(self, sample_fname, node_id):
        self.fname = sample_fname
        self.node_id = node_id
        # self.var_type = self.__compute_var_type()
        # self.var_usages = self.__compute_var_usages()



    def __compute_var_type(self):

        with open(self.fname, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            var_type = get_var_type(g, self.node_id)

        return var_type


    def __compute_var_usages(self):

        with open(self.fname, "rb") as f:
            g = Graph()
            g.ParseFromString(f.read())

            n_usages = get_var_usages(g, self.node_id)

        return n_usages


    def get_var_type(self): return self.var_type


    def get_var_usages(self): return self.var_usages




class CorpusMetaInformation():

    def __init__(self):
        self.sample_meta_inf = []


    def add_sample_inf(self, sample_inf):
        self.sample_meta_inf.append(sample_inf)


    def process_sample_inf(self):
        return None






# Pre-compute successors and predecessors for a given graph
def compute_successors_and_predecessors(graph, id_to_node):

    successors = defaultdict(list)
    predecessors = defaultdict(list)

    for edge in graph.edge:
        successors[id_to_node[edge.sourceId]].append(id_to_node[edge.destinationId])
        predecessors[id_to_node[edge.destinationId]].append(id_to_node[edge.sourceId])

    return successors, predecessors


def compute_id_dict(graph):

    id_dict = {}

    for node in graph.node:
        id_dict[node.id] = node

    return id_dict



def get_var_type(graph, sym_var_node_id):

    id_dict = compute_id_dict(graph)
    successors, predecessors = compute_successors_and_predecessors(graph, id_dict)

    id_token_nodes = [n for n in successors[sym_var_node_id] if n.type == FeatureNode.IDENTIFIER_TOKEN]

    ast_parent = None

    for id_token_node in id_token_nodes:
        for parent in predecessors[id_token_node]:
            if parent.type == FeatureNode.AST_ELEMENT and parent.contents == "VARIABLE":
                ast_parent = parent
                break

        if ast_parent != None: break

    if ast_parent == None: raise ValueError('AST_ELEMENT VARIABLE node not found...')


    fake_ast_type = [n for n in successors[ast_parent]
                     if n.type == FeatureNode.FAKE_AST and n.contents == "TYPE"][0]

    fake_ast_type_succ = list(successors[fake_ast_type])[0]

    type = [n.contents for n in predecessors[fake_ast_type_succ] if n.type == FeatureNode.TYPE][0]

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





