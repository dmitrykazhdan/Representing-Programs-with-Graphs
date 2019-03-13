from graph_pb2 import FeatureNode, FeatureEdge


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
                       FeatureNode.FAKE_AST,
                       FeatureNode.SYMBOL_TYP, FeatureNode.COMMENT_LINE,
                       FeatureNode.TYPE]

    return used_node_types
