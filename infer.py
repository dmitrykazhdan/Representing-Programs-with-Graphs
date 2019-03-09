import vocabulary_extractor
from model import model
import yaml

def infer():

  with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

  checkpoint_path = cfg['checkpoint_path']
  test_path = cfg['test_path']
  token_path = cfg['token_path']

  # Run inference
  vocabulary = vocabulary_extractor.load_vocabulary(token_path)
  m = model(mode='infer', vocabulary=vocabulary)
  m.infer(test_path, checkpoint_path=checkpoint_path)

  print("Inference ran successfully...")

infer()

