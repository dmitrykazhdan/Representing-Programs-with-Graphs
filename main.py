import tensorflow as tf
from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from dpu_utils.tfmodels import SparseGGNN
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary
from typing import List, Optional, Dict, Any
from itertools import groupby
import numpy as np
from collections import defaultdict
import os




class model():

    def __init__(self, mode, vocabulary):

        self.checkpoint_path = "/Users/AdminDK/Dropbox/Part III Modules/R252 Machine Learning " \
                               "for Programming/Project/checkpoint/train.ckpt"

        self.params = self.get_gnn_params()

        self.seq_length = 8

        self.vocabulary = vocabulary
        self.voc_size = len(vocabulary)
        self.slot_id = self.vocabulary.get_id_or_unk('<SLOT>')
        self.sos_token = self.vocabulary.get_id_or_unk('sos_token')
        self.eos_token = self.vocabulary.get_id_or_unk('eos_token')

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.batch_size = 1
        self.mode = mode

        with self.graph.as_default():
            self.embedding_size = self.params['hidden_size']
            self.placeholders = {}
            self.make_model()

            if self.mode == 'train':
                self.make_train_step()

            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            self.sess.run(init_op)


    # TODO: Add batched iteration
    # TODO: Consider other ways of handling variable-length sequences, besides padding


    def make_inputs(self):

        # Padded graph node sub-token sequences
        self.placeholders['node_token_ids'] = tf.placeholder(tf.int32, [None, self.seq_length])

        # Graph adjacency lists
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2]) for _ in range(self.params['n_edge_types'])]

        # Graph of incoming edges per type
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.params['n_edge_types']])

        # Node identifiers of all graph nodes of the target variable
        self.placeholders['slot_ids'] = tf.placeholder(tf.int32, [None], name='slot_tokens')

        #
        self.placeholders['decoder_inputs'] = tf.placeholder(shape=(self.seq_length, 1), dtype=tf.int32, name='dec_inputs')

        # Actual variable name, as a padded sequence of tokens
        self.placeholders['decoder_targets'] = tf.placeholder(dtype=tf.int32, shape=(1, self.seq_length), name='dec_targets')

        # Specify output sequence lengths
        self.placeholders['decoder_targets_length'] = tf.placeholder(shape=(1), dtype=tf.int32)

        # 0/1 matrix masking out tensor elements outside of the sequence length
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [None, None])



    def make_model(self):

        self.make_inputs()

        # Compute the embedding of input node sub-tokens
        self.embedding_encoder = tf.get_variable('embedding_encoder', [self.voc_size, self.embedding_size])
        self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_encoder, self.placeholders['node_token_ids'])

        # Average the sub-token embeddings for every node
        self.placeholders['averaged_initial_representation'] = tf.reduce_mean(self.embedding_inputs, axis=1)

        # Run graph through GGNN
        self.gnn_model = SparseGGNN(self.params)
        self.placeholders['gnn_representation'] = self.gnn_model.sparse_gnn_layer(1.0,
                                                                        self.placeholders['averaged_initial_representation'],
                                                                        self.placeholders['adjacency_lists'],
                                                                        self.placeholders['num_incoming_edges_per_type'],
                                                                        self.placeholders['num_incoming_edges_per_type'],
                                                                        {})

        # Compute average of <SLOT> usage representations
        self.placeholders['avg_representation'] = tf.expand_dims(tf.reduce_mean(tf.gather(self.placeholders['gnn_representation'],
                                                                                self.placeholders['slot_ids']), axis=0), 0)

        # Obtain output sequence by passing through a single GRU layer
        self.embedding_decoder = tf.get_variable('embedding_decoder', [self.voc_size, self.embedding_size])
        self.decoder_embedding_inputs = tf.nn.embedding_lookup(self.embedding_decoder, self.placeholders['decoder_inputs'])
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.params['hidden_size'])
        self.decoder_initial_state = self.placeholders['avg_representation']

        self.projection_layer = tf.layers.Dense(self.voc_size, use_bias=False)


        if self.mode == 'train':

            # Define training sequence decoder
            self.train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embedding_inputs,
                                                                  self.placeholders['decoder_targets_length']
                                                                  , time_major=True)

            self.train_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.train_helper,
                                                                 initial_state=self.decoder_initial_state,
                                                                 output_layer=self.projection_layer)

            self.decoder_outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(self.train_decoder)
            self.decoder_logits_train = self.decoder_outputs_train.rnn_output


        elif self.mode == 'infer':

            # Define inference sequence decoder
            start_tokens = tf.fill([self.batch_size], self.sos_token)
            end_token = self.vocabulary.get_id_or_unk(self.vocabulary.get_pad())
            max_iterations = self.seq_length * 2

            self.inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                              start_tokens=start_tokens, end_token=end_token)


            self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.inference_helper,
                                                                     initial_state=self.decoder_initial_state,
                                                                     output_layer=self.projection_layer)

            self.outputs_inference, _, _ = tf.contrib.seq2seq.dynamic_decode(self.inference_decoder,
                                                                            maximum_iterations=max_iterations)

            self.predictions = self.outputs_inference.sample_id


        else:
            raise ValueError("Invalid mode. Please specify \'train\' or \'infer\'...")


        print ("Model built successfully...")


    def make_train_step(self):

        self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders['decoder_targets'], logits=self.decoder_logits_train)
        self.train_loss = tf.reduce_sum(self.crossent * self.placeholders['target_mask'])

        # Calculate and clip gradients
        self.train_vars = tf.trainable_variables()
        self.gradients = tf.gradients(self.train_loss, self.train_vars)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 5.0)

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(0.01)
        self.train_step = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.train_vars))



    def get_gnn_params(self):

        gnn_params = {}
        gnn_params["n_edge_types"] = 8
        gnn_params["hidden_size"] = 64
        gnn_params["edge_features_size"] = {}  # Dict from edge type to feature size
        gnn_params["add_backwards_edges"] = True
        gnn_params["message_aggregation_type"] = "sum"
        gnn_params["layer_timesteps"] = [8]
        gnn_params["use_propagation_attention"] = False
        gnn_params["use_edge_bias"] = False
        gnn_params["graph_rnn_activation"] = "relu"
        gnn_params["graph_rnn_cell"] = "gru"
        gnn_params["residual_connections"] = {}  #
        gnn_params["use_edge_msg_avg_aggregation"] = False

        return gnn_params




    # def compute_edges_per_type(self, graph, incoming=True):
    #
    #     # n_nodes = ... #TODO: ensure ids are consecutive
    #
    #     # TODO: assumes node ids range from 0 to n_nodes. Ensure this assumption actually holds
    #     n_nodes, n_edge_types = 10, 5
    #     edges_matrix = np.zeros((n_nodes, n_edge_types), np.int64)
    #
    #     if incoming:
    #         for e in graph.edge: edges_matrix[e.destinationId, e.type] += 1
    #     else:
    #         for e in graph.edge: edges_matrix[e.sourceId, e.type] += 1
    #
    #     return edges_matrix



    def compute_adjacency_lists(self, edges, id_to_ind):

        adj_lists = defaultdict(list)

        for edge in edges:
            type_id = edge.type - 1
            adj_lists[type_id].append([id_to_ind[edge.sourceId], id_to_ind[edge.destinationId]])

        final_adj_lists = {edge_type: np.array(sorted(adj_list), dtype=np.int32)
                           for edge_type, adj_list in adj_lists.items()}


        print("types: ", len(final_adj_lists))

        return final_adj_lists


    def compute_initial_node_representation(self, nodes):

        max_size = self.seq_length
        padding_element = self.vocabulary.get_id_or_unk(self.vocabulary.get_pad())

        node_representations = np.array([self.vocabulary.get_id_or_unk_multiple(split_identifier_into_parts(node.contents),
                                                                           max_size, padding_element)
                                                                            for node in nodes])

        return node_representations



    def get_var_nodes(self, graph, id_to_ind):

        var_nodes = defaultdict(list)

        id_nodes = [node.id for node in graph.node if node.type == FeatureNode.IDENTIFIER_TOKEN]
        sym_nodes = [node.id for node in graph.node if node.type == FeatureNode.SYMBOL_VAR]


        # for n in graph.node:
        #     if n.type == FeatureNode.SYMBOL_VAR:
        #         print("Variable: ", n.contents)


        for edge in graph.edge:
            if edge.sourceId in sym_nodes and edge.destinationId in id_nodes:
                var_nodes[edge.sourceId].append(id_to_ind[edge.destinationId])

        return var_nodes



    def __get_id_to_int_mapping(self, nodes):

        id_to_ind = {}

        ind = 0
        for node in nodes:
            id_to_ind[node.id] = ind
            ind += 1

        return id_to_ind



    def filter_graph(self, graph):

        used_node_types = [FeatureNode.TOKEN, FeatureNode.AST_ELEMENT, FeatureNode.COMMENT_LINE,
                           FeatureNode.COMMENT_BLOCK, FeatureNode.COMMENT_JAVADOC, FeatureNode.IDENTIFIER_TOKEN,
                           FeatureNode.FAKE_AST, FeatureNode.SYMBOL, FeatureNode.SYMBOL_TYP,
                           FeatureNode.SYMBOL_VAR, FeatureNode.SYMBOL_MTH]

        used_edge_types = [FeatureEdge.NEXT_TOKEN, FeatureEdge.AST_CHILD, FeatureEdge.LAST_WRITE,
                           FeatureEdge.LAST_USE, FeatureEdge.COMPUTED_FROM, FeatureEdge.RETURNS_TO,
                           FeatureEdge.FORMAL_ARG_NAME, FeatureEdge.GUARDED_BY, FeatureEdge.GUARDED_BY_NEGATION,
                           FeatureEdge.LAST_LEXICAL_USE]


        filtered_nodes = [node for node in graph.node if node.type in used_node_types]
        filtered_node_ids = [node.id for node in filtered_nodes]

        filtered_edges = [edge for edge in graph.edge if edge.type in used_edge_types
                                                         and edge.sourceId in filtered_node_ids
                                                         and edge.destinationId in filtered_node_ids]

        return filtered_nodes, filtered_edges





    def create_sample(self, filepath):

        with open(filepath, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())

            filtered_nodes, filtered_edges = self.filter_graph(g)


            # print("Nodes: ", filtered_nodes)
            # print("Edges: ", filtered_edges)

            id_to_int = self.__get_id_to_int_mapping(filtered_nodes)
            variable_node_ids = self.get_var_nodes(g, id_to_int)
            adj_lists = self.compute_adjacency_lists(filtered_edges, id_to_int)
            node_reps = self.compute_initial_node_representation(filtered_nodes)

            samples = []


            for variable_root_id in variable_node_ids:

                node_representation = node_reps.copy()

                for variable_node_id in variable_node_ids[variable_root_id]:
                    node_representation[variable_node_id, :] = 0
                    node_representation[variable_node_id, 0] = self.slot_id


                target_mask = np.ones((self.seq_length, 1))
                variable_representation = node_reps[variable_node_id]

                for i in range(len(variable_representation)):
                    if variable_representation[i] == 0: target_mask[i] = 0




                decoder_inputs = variable_representation.copy()
                decoder_inputs = np.insert(decoder_inputs, 0, self.sos_token)[:-1]
                decoder_inputs = decoder_inputs.reshape(self.seq_length, 1)

                decoder_targets = variable_representation.copy()
                # decoder_targets = np.append(decoder_targets, self.eos_token)
                decoder_targets = decoder_targets.reshape(1, self.seq_length)

                graph_sample = {
                    self.placeholders['node_token_ids']: node_representation,
                    self.placeholders['num_incoming_edges_per_type']: np.zeros((node_representation.shape[0],
                                                                                self.params['n_edge_types']),
                                                                               dtype=np.float32),

                    self.placeholders['slot_ids']: variable_node_ids[variable_root_id],
                    self.placeholders['decoder_targets']: decoder_targets,
                    self.placeholders['decoder_inputs']: decoder_inputs,
                    self.placeholders['decoder_targets_length']: np.ones((1)) * self.seq_length,
                    self.placeholders['target_mask']: target_mask
                }



                i = 0
                for key in adj_lists:
                    graph_sample[self.placeholders['adjacency_lists'][i]] = adj_lists[key]
                    i += 1

                samples.append(graph_sample)


            return samples





    def get_graph_samples(self, dir_path):

        graph_samples = []

        for dirpath, dirs, files in os.walk(dir_path):
            for filename in files:
                if filename[-5:] == 'proto':
                    fname = os.path.join(dirpath, filename)
                    graph_samples += self.create_sample(fname)

        return graph_samples




    def train(self, path):

        graph_samples = self.get_graph_samples(path)

        with self.graph.as_default():

            for iteration in range(60):

                print("Epoch: ", iteration)
                print("----------------------------------------")

                for graph in graph_samples:
                    loss = self.sess.run([self.train_loss, self.train_step], feed_dict=graph)
                    print("Loss:", loss)

            saver = tf.train.Saver()
            saver.save(self.sess, self.checkpoint_path)



    def infer(self, path):

        graph_samples = self.get_graph_samples(path)

        with self.graph.as_default():

            saver = tf.train.Saver()
            saver.restore(self.sess, self.checkpoint_path)
            print("Model loaded successfully...")

            for graph in graph_samples:
                predictions = self.sess.run([self.predictions], feed_dict=graph)[0]
                p = [self.vocabulary.get_name_for_id(token_id) for token_id in predictions[0]]
                print(p)





def create_vocabulary(path):

    all_tokens = []

    for dirpath, dirs, files in os.walk(path):
        for filename in files:
            if filename[-5:] == 'proto':
                fname = os.path.join(dirpath, filename)

                with open(fname, "rb") as f:
                    g = Graph()
                    g.ParseFromString(f.read())

                    for n in g.node:
                        all_tokens += split_identifier_into_parts(n.contents)


    all_tokens = list(set(all_tokens))
    all_tokens.append('<SLOT>')
    all_tokens.append('sos_token')
    all_tokens.append('eos_token')
    all_tokens.sort()

    vocabulary = Vocabulary.create_vocabulary(all_tokens, max_size=1000, count_threshold=0,
                                                   add_unk=True, add_pad=True)

    return vocabulary





def main(path):

  vocabulary = create_vocabulary(path)

  print("Vocabulary: ", vocabulary)

  m = model('train', vocabulary)
  m.train(path)

  m = model('infer', vocabulary)
  m.infer(path)





main("/Users/AdminDK/Desktop/sample_graphs")





'''
Nodes:
FAKE_AST, SYMBOL, SYMBOL_TYP, SYMBOL_VAR, SYMBOL_MTH, COMMENT_LINE, COMMENT_BLOCK, COMMENT_JAVADOC ?

Used Node Types: TOKEN, AST_ELEMENT, IDENTIFIER_TOKEN


Edges:
ASSOCIATED_TOKEN, NONE, COMMENT

Used Edge Types: NEXT_TOKEN, AST_CHILD, LAST_WRITE, LAST_USE, COMPUTED_FROM, RETURNS_TO, FORMAL_ARG_NAME, GUARDED_BY,
GUARDED_BY_NEGATION, LAST_LEXICAL_USE, 
'''






