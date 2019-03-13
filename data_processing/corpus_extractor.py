import os
import yaml
from random import shuffle
from shutil import copyfile



def split_samples():

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    corpus_path = cfg['corpus_path']
    train_path  = cfg['train_path']
    val_path    = cfg['val_path']
    test_path   = cfg['test_path']

    f_names = []
    ignore = ("Test.java.proto", "TestCase.java.proto", "Tests.java.proto") # Ignore test cases
    max_size_mb = 100          # maximum file size in MB
    min_size_mb = 0.05


    # Extract all filenames from corpus folders
    for dirpath, dirs, files in os.walk(corpus_path):
        for filename in files:
            if filename.endswith('proto') and not filename.endswith(ignore):

                fname = os.path.join(dirpath, filename)

                f_size_mb = os.path.getsize(fname) / 1000000

                if f_size_mb < max_size_mb and f_size_mb > min_size_mb:
                    fname = os.path.join(dirpath, filename)
                    f_names.append(fname)



    # Copy subset of samples into training/validation/testing directories
    n_samples = len(f_names)
    n_train_and_val = round(n_samples * 0.85)
    n_train = round(n_train_and_val * 0.85)

    shuffle(f_names)

    train_samples = f_names[:n_train]
    val_samples = f_names[n_train:n_train_and_val]
    test_samples = f_names[n_train_and_val:n_samples]

    copy_samples(train_samples, train_path)
    copy_samples(val_samples, val_path)
    copy_samples(test_samples, test_path)



def copy_samples(sample_names, base_path):

    for src in sample_names:
        dst = os.path.join(base_path, os.path.basename(src))
        copyfile(src, dst)




split_samples()
