import os
from graph_pb2 import Graph
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary
import pickle


# Create vocabulary using the training corpus
def create_vocabulary_from_corpus(training_corpus_path, output_token_path=None):

    all_sub_tokens = []

    # Extract all subtokens from all nodes in all graphs in the corpus
    for dirpath, dirs, files in os.walk(training_corpus_path):
        for filename in files:
            if filename[-5:] == 'proto':
                fname = os.path.join(dirpath, filename)

                with open(fname, "rb") as f:
                    g = Graph()
                    g.ParseFromString(f.read())

                    for n in g.node:
                        all_sub_tokens += split_identifier_into_parts(n.contents)

    all_sub_tokens = list(set(all_sub_tokens))

    # Add special sequence-processing tokens
    all_sub_tokens.append('<SLOT>')
    all_sub_tokens.append('sos_token')
    all_sub_tokens.append('eos_token')
    all_sub_tokens.sort()


    # Save all extracted subtokens
    if output_token_path != None:
        with open(output_token_path, "wb") as fp:
            pickle.dump(all_sub_tokens, fp)

    return __create_voc_from_tokens(all_sub_tokens)



def load_vocabulary(token_path):

    if not os.path.isfile(token_path):
        raise ValueError("Error. No file found...")

    with open(token_path, "rb") as fp:
        sub_tokens = pickle.load(fp)

    return __create_voc_from_tokens(sub_tokens)



def __create_voc_from_tokens(all_sub_tokens):

    vocabulary = Vocabulary.create_vocabulary(all_sub_tokens, max_size=100000, count_threshold=0,
                                                   add_unk=True, add_pad=True)

    return vocabulary