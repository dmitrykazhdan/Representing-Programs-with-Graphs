import os
import yaml
from random import shuffle
from shutil import copyfile



def get_train_and_test():

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    corpus_path = cfg['corpus_path']
    train_path  = cfg['train_path']
    test_path   = cfg['test_path']

    f_names = []
    ignore = ("Test.java.proto", "TestCase.java.proto", "Tests.java.proto") # Ignore test cases
    max_size_mb = 5          # maximum file size in MB


    # Extract all filenames from corpus folders
    for dirpath, dirs, files in os.walk(corpus_path):
        for filename in files:
            if filename.endswith('proto') and not filename.endswith(ignore):

                fname = os.path.join(dirpath, filename)

                f_size = os.path.getsize(fname) / 1000000

                if f_size < max_size_mb:
                    fname = os.path.join(dirpath, filename)
                    f_names.append(fname)



    # Copy subset of samples into training/testing directories
    n_samples = 100
    n_train = round(n_samples * 0.8)
    shuffle(f_names)

    train_samples = f_names[:n_train]
    test_samples = f_names[n_train:n_samples]


    for src in train_samples:
        dst = os.path.join(train_path, os.path.basename(src))
        copyfile(src, dst)


    for src in test_samples:
        dst = os.path.join(test_path, os.path.basename(src))
        copyfile(src, dst)



get_train_and_test()
