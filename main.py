import vocabulary_extractor
from model import model
import yaml

def main():

  with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

  checkpoint_path = cfg['checkpoint_path']
  train_path = cfg['train_path']
  test_path = cfg['test_path']
  token_path = cfg['token_path']


  # Run training
  vocabulary = vocabulary_extractor.create_vocabulary_from_corpus(train_path, token_path)
  print("Constructed vocabulary...")

  m = model(mode='train', vocabulary=vocabulary, checkpoint_path=checkpoint_path)
  n_train_epochs = 100
  m.train(train_path, n_train_epochs)


  # Run inference
  vocabulary = vocabulary_extractor.load_vocabulary(token_path)
  m = model(mode='infer', vocabulary=vocabulary, checkpoint_path=checkpoint_path)
  m.infer(test_path)


main()

