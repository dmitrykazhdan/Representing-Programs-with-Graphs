# Representing Programs with Graphs

This project re-implements the _VarNaming_ task model described in the _Learning to Represent Programs with Graphs_ paper 
(which can be found [here](https://ml4code.github.io/publications/allamanis2018learning/), along with other relevant resources), 
which can predict the name of a variable based on it's usage.

Furthermore, this project includes functionality for applying the _VarNaming_ model to the _MethodNaming_ task 
(predicting the name of a method from it's usage or definition). 


## Setup 
### Prerequisites

Ensure you have the following packages installed:

- numpy
- pyYAML
- tensorflow-gpu
- dpu_utils


### Dataset Format

The corpus pre-processing functions are designed to work with _.proto_ 
graph files, which can be extracted from program source code using the feature
extractor available [here](https://github.com/acr31/features-javac).




### Dataset Parsing

Once you have obtained a corpus of .proto graph files, it is possible
to use the _corpus_extractor.py_ file located in the _data_processing_ folder.

- Create empty directories for training, validation and testing datasets
- Specify their paths, as well as the corpus path, in the
_config.yml_ file:
```python
corpus_path: "path-to-corpus"
train_path: "path-to-train-data-output"
val_path: "path-to-val-data-output"
test_path: "path-to-test-data-output"
```
- Run _corpus_extractor.py_ 

```python
python3 path-to-repository/data_processing/corpus_extractor.py
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
- Specify the token path 
(where the extracted token vocabulary will be saved)
and the checkpoint path (where the model checkpoint will be saved) in the _config.yml_ file:
```python
checkpoint_path: "path-to-checkpoint-folder/train.ckpt"
token_path: "path-to-vocabulary-file/tokens.txt"
```
- Run _train.py_

```python
python3 path-to-repository/train.py
```


### Inference

In order to use the model for inference:

- Prepare the test dataset directory
as described in the _Dataset Parsing_ section above
- Specify its paths in the _config.yml_ file:
```python
test_path: "path-to-test-data"
```
- Specify the token path 
(where the extracted token vocabulary will be loaded from)
and the checkpoint path (where the trained model will be loaded from) in the _config.yml_ file:
```python
checkpoint_path: "path-to-checkpoint-folder/train.ckpt"
token_path: "path-to-vocabulary-file/tokens.txt"
```
- Run _infer.py_

```python
python3 path-to-repository/infer.py
```


### In-depth inference

In order to use the model for inference, as well as extra sample information
(such as usage information and type information):

- Prepare the test dataset directory
as described in the _Dataset Parsing_ section above
- Specify its paths in the _config.yml_ file:
```python
test_path: "path-to-test-data"
```
- Specify the token path 
(where the extracted token vocabulary will be loaded from)
and the checkpoint path (where the trained model will be loaded from) in the _config.yml_ file:
```python
checkpoint_path: "path-to-checkpoint-folder/train.ckpt"
token_path: "path-to-vocabulary-file/tokens.txt"
```
- Run _detailed_infer.py_

```python
python3 path-to-repository/detailed_infer.py
```

For more details on the extra information computed, see the report 
available [here](link_TBC)



### Task Selection
The type of task you want the model to run with can be specified by passing 
appropriate input arguments as follows:

- To run training/inference using the VarNaming task (computing variable usage information)
no input arguments are required
- To run training/inference using the MethodNaming usage task (computing method usage information)
add a _mth_usage_ argument when calling the script
- To run training/inference using the MethodNaming definition task (computing method body information)
add a _mth_def_ argument when calling the script

For example, in order to train the model for the MethodNaming task using 
definition information, the script call will be the following:

```python
python3 path-to-repository/train.py mth_definition
```
