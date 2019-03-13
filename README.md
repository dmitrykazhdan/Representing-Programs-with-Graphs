# Representing Programs with Graphs

This project re-implements the _VarNaming_ task model described in the _Learning to Represent Programs with Graphs_ paper 
(which can be found [here](https://ml4code.github.io/publications/allamanis2018learning/), along with other relevant resources), 
which can predict the name of a variable based on it's usage.

Furthermore, this project includes functionality for applying the _VarNaming_ model to the _MethodNaming_ task 
(predicting the name of a method from it's usage and/or definition). 


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
- Specify their paths, together well as the corpus path, in the
_config.yml_ file
- Run _corpus_extractor.py_ 

```python
python3 path-to-repository/data_processing/corpus_extractor.py
```

This will extract all samples from the corpus, randomly shuffle them,
split them into train/val/test partitions, and copy these partitions into the specified
folders.



## Usage

### Training 

In order to train the model:

- Prepare training and validation dataset directories
- Specify their paths in the _config.yml_ file
- Specify paths where the extracted token vocabulary
and model checkpoints will be saved in the _config.yml_ file
- Run _train.py_

```python
python3 path-to-repository/train.py
```


### Inference

In order to use the model for inference:

- Prepare the test dataset directory
- Specify its paths in the _config.yml_ file
- Specify paths to the extracted token vocabulary
and model checkpoints saved during training in the _config.yml_ file
- Run _infer.py_

```python
python3 path-to-repository/infer.py
```


### In-depth inference

In order to use the model for inference, as well as extra sample information
(such as usage information and type information):

- Prepare the test dataset directory
- Specify its paths in the _config.yml_ file
- Specify paths to the extracted token vocabulary
and model checkpoints saved during training in the _config.yml_ file
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
