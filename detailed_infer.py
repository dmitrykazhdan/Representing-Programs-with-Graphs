from utils import vocabulary_extractor
from model.model import Model
import yaml
import sys
from utils.arg_parser import parse_input_args


def detailed_inference(task_id):

  with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

  checkpoint_path = cfg['checkpoint_path']
  train_path = cfg['train_path']
  test_path = cfg['test_path']
  token_path = cfg['token_path']

  # Run inference
  vocabulary = vocabulary_extractor.load_vocabulary(token_path)
  m = Model(mode='infer', task_id=task_id, vocabulary=vocabulary)
  m.metrics_on_seen_vars(train_path, test_path, checkpoint_path=checkpoint_path)

  print("Inference ran successfully...")


detailed_inference(0)

