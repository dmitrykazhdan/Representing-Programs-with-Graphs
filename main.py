import tensorflow as tf
from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from dpu_utils.tfmodels import SparseGGNN
import numpy as np
from collections import defaultdict
import os
import vocabulary_extractor, graph_preprocessing
import matplotlib.pyplot as plt
from random import shuffle
from shutil import copyfile
import math



def main():

  # Training:
  n_train_epochs = 60
  vocabulary = vocabulary_extractor.create_vocabulary_from_corpus(train_path, token_path)
  m = model('train', vocabulary)
  m.train(train_path, n_train_epochs)


  # Inference
  vocabulary = vocabulary_extractor.load_vocabulary(token_path)
  m = model('infer', vocabulary)
  test_acc = m.infer(test_path)








def get_train_and_test(corpus_path, train_path, test_path):

    f_names = []

    for dirpath, dirs, files in os.walk(corpus_path):
        for filename in files:
            if filename[-5:] == 'proto':
                fname = os.path.join(dirpath, filename)
                f_names.append(fname)

    n_samples = 100
    n_train = round(n_samples * 0.8)
    shuffle(f_names)

    train_samples = f_names[:n_train]
    test_samples = f_names[n_train:n_samples]

    print(train_samples)

    for src in train_samples:
        dst = os.path.join(train_path, os.path.basename(src))
        copyfile(src, dst)


    for src in test_samples:
        dst = os.path.join(test_path, os.path.basename(src))
        copyfile(src, dst)


corpus_path = "/Users/AdminDK/Dropbox/Part III Modules/R252 Machine Learning for Programming/Practicals/corpus/r252-corpus-features"
checkpoint_path = "/Users/AdminDK/Dropbox/Part III Modules/R252 Machine Learning for Programming/Project/checkpoint/train.ckpt"
train_path = "/Users/AdminDK/Desktop/train_graphs"
test_path = "/Users/AdminDK/Desktop/test_graphs"
token_path = "/Users/AdminDK/Desktop/tokens.txt"


#get_train_and_test(corpus_path, train_path, test_path)
main()


# for main_epoch in epochs:
#     main()
#plt.plot(epochs, main_accuracies)
# plt.scatter(epochs, main_accuracies, marker='x', color='red')
# plt.xlabel('Training Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training Data Accuracy')
# plt.show()