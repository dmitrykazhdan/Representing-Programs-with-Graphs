from utils import vocabulary_extractor
from model.model import Model
import yaml
import sys
from utils.arg_parser import parse_input_args

def infer(task_id):

  with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

  checkpoint_path = cfg['checkpoint_path']
  test_path = cfg['test_path']
  token_path = cfg['token_path']


  vocabulary = vocabulary_extractor.load_vocabulary(token_path)
  m = Model(mode='infer', task_id=task_id, vocabulary=vocabulary)

  m.infer(corpus_path=test_path, checkpoint_path=checkpoint_path)
  print("Inference ran successfully...")



args = sys.argv[1:]
task_id = parse_input_args(args)

infer(task_id)

