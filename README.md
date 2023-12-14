# Data Valuation Platform
![python version](https://img.shields.io/badge/python-v.3.8-blue)
![license](https://img.shields.io/badge/license-MIT-orange)

## Repository Structure
- [DataValuationPlatform](DataValuationPlatform): This folder contains the codebase for the DataValuationPlatform Package. This package allows the easy application of Data Valuation Methods with predefined Data Loader and Model classes as well as implemented applications such as false positive detection, active learning, and undersampling.
- [Datasets](Datasets): This folder contains the 25 preprocessed datasets used for the false positive detection application and the active learning application as .csv files, split into training and validation set.
- [Experiments](Experiments): This folder contains all scripts used to generate the results presented in our publication, as well as the results themselves.


## DataValuationPlatform
This package can be used to implement a range of data valuation methods. It has the following features:
### Features
- **Data Loading and Preprocessing**: The platform includes an HTS Data Processor that allows easy preprocessing of pubchem datasets. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/preprocessor.py)
- **Model Integration**: The platform supports various data valuation models, each offering unique approaches to data valuation.
- **Ready-to-use applications**: Applications such as active learning, false positive detection, and importance undersampling are implemented for all data valuation models and ready to use.

The package contains the data valuation models descriped in our manuscript:
### Data Valuation Models
1. **CatBoost Model**: An implementation of the CatBoost algorithm, known for handling categorical data efficiently. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/catboost/CatBoost_model.py)
2. **DVRL Model**: Integrates the DVRL (Data Valuation using Reinforcement Learning) approach for data valuation. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/dvrl/DVRL_model.py)
3. **KNN Shapley Model**: Applies the KNN Shapley method, a technique based on Shapley values, for assessing data influence. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/knn_shapley/KNN_Shapley_model.py)
4. **TracIn Model**: Applies the TracIn method, calculating sample influence by tracing gradient descent. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/knn_shapley/TracIn_model.py)
5. **MVSA Model**: Implements the MVSA (Most Valuable Subset Analysis) for evaluating data subsets. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/mvsa/MVSA_model.py)

### Tutorial
The following tutorial shows how to load some of the datasets included in this repository into a jupyter notebook, calculate molecular descriptors, and use one of the data valuation methods for false positive prediction

```python
from DataValuationPlatform import HTSDataPreprocessor, MVSA, TracIn, CatBoost, DVRL

#create a preprocessor object and load the datasets you are interested in (e.g. the preprocessed datasets supplied in this repository by using their names)
preprocessor = HTSDataPreprocessor(["GPCR_3", "GPCR_2", "GPCR"])
preprocessor.load_preprocessed_data()

#calculate their molecular descriptors (currently implemented are ECFPs, a set of 208 RDKit descriptors, and SMILES)
preprocessor.create_descriptors(descriptor_type = "ecfp")

# create dataset objects for each dataset, which contain their train and test sets, molecular descriptors, labels
dataset_gpcr3 = preprocessor.get_dataset("GPCR_3")
dataset_gpcr2 = preprocessor.get_dataset("GPCR_2")

#create a data valuation model
mvsa_model = MVSA()

#you can either use these models just for calculating importance scores for a dataset
gpcr3_influence_scores = mvsa_model.calculate_influence(dataset_gpcr3)

#or apply one of the applications explained in the paper

#false positive prediction
gpcr3_false_positives_mvsa_results,gpcr3_mvsa_logs = mvsa_model.apply_false_positive_identification(dataset = dataset_gpcr3, replicates = 3)

#active learning
gpcr3_active_learning_mvsa_results = mvsa_model.apply_active_learning(dataset = dataset_gpcr3, step_size = 1, steps = 6, regression_function = "gpr", sampling_function = "greedy")

#importance undersampling
gpcr3_undersampling_mvsa_results = mvsa_model.apply_undersampling(dataset = dataset_gpcr3, steps = 19)
```

### Prerequisites
The platform currently supports Python 3.8. Some required packages are not included in the pip install: 
- [Tensorflow](https://www.tensorflow.org/) (2.4.0)
- [Datascope](https://pypi.org/project/datascope/0.0.10/) (0.0.10)
- 

### Installation
The DataValuationPlatform will be installable using pip. Alternatively, you can clone this repository manually. 
The KNN Shapley model is not available in the pip install package due to incompatibilites with the remaining platform (datascope uses numpy version 1.24.2, the remaining packages uses 1.19.2). 


