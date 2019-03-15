# Representing Programs with Graphs

This project re-implements the _VarNaming_ task model described in the  paper 
[_Learning to Represent Programs with Graphs_](https://ml4code.github.io/publications/allamanis2018learning/), 
which can predict the name of a variable based on it's usage.

Furthermore, this project includes functionality for applying the _VarNaming_ model to the _MethodNaming_ task 
(predicting the name of a method from it's usage or definition). 


## Setup 
### Prerequisites

Ensure you have the following packages installed 
(these can all be installed with pip3):

- numpy
- pyYAML
- tensorflow-gpu (or tensorflow)
- dpu_utils
- protobuf


### Dataset Format

The corpus pre-processing functions are designed to work with _.proto_ 
graph files, which can be extracted from program source code using the feature
extractor available [here](https://github.com/acr31/features-javac).




### Dataset Parsing

Once you have obtained a corpus of .proto graph files, it is possible
to use the _corpus_extractor.py_ file located in the _data_processing_ folder.

- Create empty directories for training, validation and test datasets
- Specify their paths, as well as the corpus path, in the
_config.yml_ file:
```python
corpus_path: "path-to-corpus"
train_path: "path-to-train-data-output"
val_path: "path-to-val-data-output"
test_path: "path-to-test-data-output"
```
- Navigate into the repository directory
- Run _corpus_extractor.py_:

```python
python3 ./data_processing/corpus_extractor.py
```

This will extract all samples from the corpus, randomly shuffle them,
split them into train/val/test partitions, and copy these partitions into the specified
train, val and test folders.



## Usage

### Training 

In order to train the model:

- Prepare training and validation dataset directories, 
as described in the _Dataset Parsing_ section above
- Specify their paths in the _config.yml_ file:
```python
train_path: "path-to-train-data"
val_path: "path-to-val-data"
```
- Specify the token file path 
(where the extracted token vocabulary will be saved)
and the checkpoint folder path (where the model checkpoint will be saved) in the _config.yml_ file 
(note the fixed specification of the 'train.ckpt' file):
```python
checkpoint_path: "path-to-checkpoint-folder/train.ckpt"
token_path: "path-to-vocabulary-txt-file"
```
- Navigate into the repository directory
- Run _train.py_:

```python
python3 ./train.py
```


### Inference

In order to use the model for inference:

- Prepare the test dataset directory
as described in the _Dataset Parsing_ section above
- Specify it's path in the _config.yml_ file:
```python
test_path: "path-to-test-data"
```
- Specify the token file path 
(where the extracted token vocabulary will be loaded from)
and the checkpoint path (where the trained model will be loaded from) in the _config.yml_ file:
```python
checkpoint_path: "path-to-checkpoint-folder/train.ckpt"
token_path: "path-to-vocabulary-txt-file"
```
- Navigate into the repository directory
- Run _infer.py_:

```python
python3 ./infer.py
```


### Detailed inference

In order to use the model for inference, 
as well as for computing extra sample information
(including variable usage information and type information):

- Prepare the test dataset directory
as described in the _Dataset Parsing_ section above
- Specify it's path in the _config.yml_ file:
```python
test_path: "path-to-test-data"
```
- Specify the token file path 
(where the extracted token vocabulary will be loaded from)
and the checkpoint path (where the trained model will be loaded from) in the _config.yml_ file:
```python
checkpoint_path: "path-to-checkpoint-folder/train.ckpt"
token_path: "path-to-vocabulary-txt-file"
```
- Navigate into the repository directory
- Run _detailed_infer.py_

```python
python3 ./detailed_infer.py
```



### MethodNaming Task Selection
The type of task you want the model to run can be specified by passing 
appropriate input arguments as follows:

- To run training/inference using the VarNaming task (computing variable usage information)
no input arguments are required
- To run training/inference using the MethodNaming usage task (computing method usage information)
add the string "_mth_usage_" as an input argument when calling the scripts
- To run training/inference using the MethodNaming definition task (computing method body information)
add the string "_mth_def_" as an input argument when calling the scripts

For example, in order to train the model for the MethodNaming task using 
definition information, the script call will be the following:

```python
python3 ./train.py mth_def
```

Similarly, for running inference using the MethodNaming definition task,
the script call will be the following:
```python
python3 ./infer.py mth_usage
```

### Loading Saved Models

The _saved_models_ directory includes pre-trained models, which can
be used to run inference directly, without any training. 
The paths to the saved checkpoint and vocabulary files need to be specified
in the _config.yml_ file 
in the usual way, as described in the "Inference" section above.





## Files/Directories

- data_processing: includes code for processing graph samples and corpus files
- model: includes the implementation of the VarNaming model
- saved_models: pre-trained models for the VarNaming and MethodNaming tasks
- utils: auxiliary code implementing various functionality, such as input 
argument parsing and vocabulary extraction
- train.py, infer.py, detailed_infer.py: files for running training and inference
using the model, as described in the previous sections
- config.yml: configuration file storing string properties
- graph_pb2.py: used for parsing .proto sample files
