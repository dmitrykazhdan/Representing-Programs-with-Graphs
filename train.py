from utils import vocabulary_extractor
from model.model import Model
import yaml

def train():

  with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

  checkpoint_path = cfg['checkpoint_path']
  train_path = cfg['train_path']
  val_path   = cfg['val_path']
  token_path = cfg['token_path']


  vocabulary = vocabulary_extractor.create_vocabulary_from_corpus(train_path, token_path)
  print("Constructed vocabulary...")

  m = Model(mode='train', vocabulary=vocabulary)
  n_train_epochs = 40

  m.train(train_path=train_path, val_path=val_path, n_epochs=n_train_epochs, checkpoint_path=checkpoint_path)
  print("Model trained successfully...")


train()

