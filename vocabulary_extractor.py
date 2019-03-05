import os
from graph_pb2 import Graph
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary
import pickle
import graph_preprocessing


# Create vocabulary using the training corpus
def create_vocabulary_from_corpus(training_corpus_path, output_token_path=None):

    all_sub_tokens = []
    node_types = graph_preprocessing.get_used_nodes_type()

    # Extract all subtokens from all nodes in all graphs in the corpus
    for dirpath, dirs, files in os.walk(training_corpus_path):
        for filename in files:
            if filename.endswith('proto'):
                fname = os.path.join(dirpath, filename)

                with open(fname, "rb") as f:
                    g = Graph()
                    g.ParseFromString(f.read())

                    for n in g.node:
                        if n.type in node_types:
                            all_sub_tokens += split_identifier_into_parts(n.contents)

    all_sub_tokens = list(set(all_sub_tokens))

    # Add special sequence-processing tokens
    all_sub_tokens.append('<SLOT>')
    all_sub_tokens.append('sos_token')
    all_sub_tokens.sort()

    vocabulary = __create_voc_from_tokens(all_sub_tokens)

    # Save all extracted subtokens
    if output_token_path != None:
        with open(output_token_path, "wb") as fp:
            pickle.dump(vocabulary, fp)

    return vocabulary



def load_vocabulary(token_path):

    if not os.path.isfile(token_path):
        raise ValueError("Error. File not found...")

    with open(token_path, "rb") as fp:
        vocabulary = pickle.load(fp)

    return vocabulary



def __create_voc_from_tokens(all_sub_tokens):

    vocabulary = Vocabulary.create_vocabulary(all_sub_tokens, max_size=100000, count_threshold=1,
                                                   add_unk=True, add_pad=True)

    return vocabulary