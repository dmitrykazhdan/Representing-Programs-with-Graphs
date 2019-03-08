import vocabulary_extractor
from model import model
import yaml

def train():

  with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

  checkpoint_path = cfg['checkpoint_path']
  train_path = cfg['train_path']
  val_path   = cfg['val_path']
  token_path = cfg['token_path']


  # Run training
  vocabulary = vocabulary_extractor.create_vocabulary_from_corpus(train_path, token_path)
  print("Constructed vocabulary...")

  m = model(mode='train', vocabulary=vocabulary, checkpoint_path=checkpoint_path)
  n_train_epochs = 150
  m.train(train_path, val_path, n_train_epochs)
  print("Model trained successfully...")

train()

