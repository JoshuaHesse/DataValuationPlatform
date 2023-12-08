# Data Valuation Platform

## Overview
The Data Valuation Platform is a comprehensive framework designed to evaluate and quantify the influence of data in various machine learning models. This platform integrates multiple algorithms and techniques to provide a robust analysis of data value.

## Features
- **Data Loading and Preprocessing**: The platform includes a HTS Data Processor that allows easy preprocessing of pubchem datasets. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/preprocessor.py)
- **Model Integration**: The platform supports various data valuation models, each offering unique approaches to data valuation.
- **Ready-to-use applications**: Applications such as active learning, false positive detection, and importance undersampling are implemented for all data valuation models and ready to use.
- **Customizable Parameters**: Offers flexibility in configuring model parameters to suit different datasets and requirements.



## Data Valuation Models
1. **CatBoost Model**: An implementation of the CatBoost algorithm, known for handling categorical data efficiently. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/catboost/CatBoost_model.py)
2. **DVRL Model**: Integrates the DVRL (Data Valuation using Reinforcement Learning) approach for data valuation. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/dvrl/DVRL_model.py)
3. **KNN Shapley Model**: Applies the KNN Shapley method, a technique based on Shapley values, for assessing data influence. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/knn_shapley/KNN_Shapley_model.py)
4. **TracIn Model**: Applies the TracIn method, calculating sample influence by tracing gradient descent. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/knn_shapley/TracIn_model.py)
5. **MVSA Model**: Implements the MVSA (Most Valuable Subset Analysis) for evaluating data subsets. [View Code](https://github.com/JoshuaHesse/DataValuationPlatform/blob/master/DataValuationPlatform/models/mvsa/MVSA_model.py)

## Installation
Provide instructions on how to install and set up your project. This might include steps to clone the repository, install dependencies, and any necessary configuration.

## Usage
Include examples or a guide on how to use the platform. You can provide code snippets or command-line instructions for users to follow.

## Contributing
Encourage contributions by providing guidelines on how contributors can get involved. Include instructions for submitting pull requests, reporting bugs, or suggesting enhancements.

## License
Specify the license under which your project is released.

## Contact
Provide your contact information or additional resources where users can get support or learn more about your project.
