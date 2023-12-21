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
#### Dataset Loading
```python
#1. you can load the preinstalled datasets used in this publication via names
preprocessor = HTSDataPreprocessor(["GPCR_3", "GPCR_2", "GPCR"])
preprocessor.load_preprocessed_data()
preprocessor.create_descriptors(descriptor_type = "ecfp")

dataset_gpcr3 = preprocessor.get_dataset("GPCR_3")
dataset_gpcr2 = preprocessor.get_dataset("GPCR_2")
dataset_gpcr = preprocessor.get_dataset("GPCR_")

#2. You can add pubchem assay combinations that combine a primary and a confirmatory assay by downloading the raw files and adding
#the dataset to the existing collection (here example with made up aids)
preprocessor = HTSDataPreprocessor([])
preprocessor.add_dataset_by_AID(codename = "MadeUpAssayName", primary_AID = "001",confirmatory_AID= "002")
preprocessor.add_dataset_by_AID(codename = "MadeUpAssayName2", primary_AID = "003",confirmatory_AID= "004")
preprocessor.preprocess_data(path_to_raw="Path/To/Raw_data/")
preprocessor.create_descriptors("ecfp")

dataset_MadeUpAssayName = preprocessor.get_dataset("MadeUpAssayName")
dataset_MadeUpAssayName2 = preprocessor.get_dataset("MadeUpAssayName2")

3. you can add your own data directly as a custom dataset:
preprocessor = HTSDataPreprocessor([])
preprocessor.create_custom_dataset(
    dataset_name = CustomDataset,
    training_set_smiles=train_smiles,
    training_set_labels=train_labels,
    training_set_confirmatory_labels=train_confirmatory_labels) #this is only necessary for the false positive identification
preprocessor.create_descriptors("ecfp")

datasetCustomDataset = preprocessor.get_dataset("CustomDataset")
```
#### Model usage
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
## Experiments
The files included here are the preprocessing scripts:
- create the Dataset files from the raw files downloaded from pubchem [cleanup_pipeline](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/Experiments/Scripts/cleanup_pipeline_jh.py)
- create molecular descriptors from Dataset files [descr_export_pipeline_jh](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/Experiments/Scripts/descr_export_pipeline_jh.py)
- create molecular descriptors for the moldata dataset [mol_data_descr_export_pipeline_jh](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/Experiments/Scripts/mol_data_descr_export_pipeline_jh.py)
as well as the scripts and utility files used to perform the experiments shown in the paper
- [FalsePositivePrediction] (https://github.com/JoshuaHesse/DataValuationPlatform/tree/master/Experiments/Scripts/FalsePositivePrediction) This folder contains the [eval_pipeline](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/Experiments/Scripts/FalsePositivePrediction/eval_pipeline_jh.py) used for the false positive prediction application as well as the necessary utility files
- [ActiveLearning](https://github.com/JoshuaHesse/DataValuationPlatform/tree/master/Experiments/Scripts/ActiveLearning) contains the [active_learning_pipeline](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/Experiments/Scripts/ActiveLearning/active_learning_pipeline_jh.py) used for the active learning application as well as the necessary utility files
- [Undersampling](https://github.com/JoshuaHesse/DataValuationPlatform/tree/master/Experiments/Scripts/Undersampling) contains the [undersampling_pipeline](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/Experiments/Scripts/Undersampling/undersampling_pipeline_jh.py) used for the undersampling application and the necessary utility files

### Usage
In order to reproduce the results, you need to first create the molecular descriptors for the datasets you are interested in using the [descr_export_pipeline_jh](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/Experiments/Scripts/descr_export_pipeline_jh.py). 
```bash
cd DataValuationPlatform/Experiments/Scripts
python3 --dataset all --representation ECFP
```
#### False Positive Prediction
Here is an example of how to use the [eval_pipeline](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/Experiments/Scripts/FalsePositivePrediction/eval_pipeline_jh.py) to test the MVS-A, Tracin, and Catboost false and true positive prediction performance on one dataset, using PAINS fragment filters and the Score method as benchmarks with 5 replicates:
```bash
cd DataValuationPlatform/Experiments/Scripts/FalsePositivePrediction
python3 eval_pipeline_jh.py --dataset GPCR_3 --knn no --dvrl no --tracin yes --mvs_a yes --catboost yes --score yes --fragment_filter yes --representation ECFP --replicates 5 --filename output --log_predictions yes --environment others
```
more information is given in the [FalsePositivePrediction](https://github.com/JoshuaHesse/DataValuationPlatform/tree/master/Experiments/Scripts/FalsePositivePrediction) folder.

#### Active Learning
Here is an example of how to use [active_learning_pipeline](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/Experiments/Scripts/ActiveLearning/active_learning_pipeline_jh.py) on all using the standard parameters on all datasets
```bash
cd DataValuationPlatform/Experiments/Scripts/ActiveLearning
python3 active_learning_pipeline_jh.py 
```
#### Undersampling
This part of the project was done using the [MolData](https://github.com/LumosBio/MolData) benchmark instead of our curated dataset group. To reproduce this, clone the moldata benchmark into this folder first and calculate the molecular descriptors
```bash
cd DataValuationPlatform/Experiments/Scripts
git clone https://github.com/LumosBio/MolData
python3 mol_data_descr_export_pipeline_jh.py --group_type disease --dataset_group aging --representation ECFP
cd DataValuationPlatform/Experiments/Scripts
python3 undersampling_pipeline_jh.py --dataset_group aging --influence MVSA --replicates 5
```

### Prerequisites
The platform currently supports Python 3.8. Some required packages are not included in the pip install: 
- [Tensorflow](https://www.tensorflow.org/) (2.4.0)
- [Datascope](https://pypi.org/project/datascope/0.0.10/) (0.0.10)
- 

### Installation
The DataValuationPlatform will be installable using pip. Alternatively, you can clone this repository manually. 
The KNN Shapley model is not available in the pip install package due to incompatibilites with the remaining platform (datascope uses numpy version 1.24.2, the remaining packages uses 1.19.2). 


