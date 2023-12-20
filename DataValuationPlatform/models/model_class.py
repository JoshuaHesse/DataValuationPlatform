#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:33:02 2023

@author: joshuahesse
"""
import numpy as np
from typing import List, Tuple, Optional
import sys
import pandas as pd
from rdkit import Chem
import os
from DataValuationPlatform.models.utils.utils_jh import *
from shutil import rmtree
import time
import copy
from sklearn.metrics import average_precision_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import lightgbm
import DataValuationPlatform.models.utils.regression_functions as rf
import DataValuationPlatform.models.utils.sampling_functions as sf
from DataValuationPlatform.models.utils import custom_kernels
from sklearn import preprocessing
import random
from DataValuationPlatform.models.preprocessor import HTSDataPreprocessor


def transform_data(
            X_train: np.ndarray,
            y_train: np.ndarray
            )-> Tuple[np.ndarray, np.ndarray, preprocessing._data.StandardScaler]:
    """
    

    Parameters
    ----------
    X_train : np.array, (M,D)
        vectorized molecular representations with M molecules and D dimensions
    y_train : np.array (M,1)
        M labels for corresponding molecules, e.g. influence scores

    Returns
    -------
    X_train_scaled : np.array, (M,D)
        scaled vectorized molecular representations with M molecules and D dimensions
    y_train_scaled : np.array (M,1)
        M scaled labels for corresponding molecules, e.g. influence scores
    y_scaler: sklearn.preprocessing._data.StandardScaler
        scaler used for y_scaling; can later be used to reverse_transform 
        y_values after e.g. regression to work with original influence scales

    ------
    
    This function transforms x_values and y_labels via StandardScalers. Within 
    this file, only the y_value transformation is really used because the ECFP
    x_values are binary and need no scaling. The RDKIT x_values read from the 
    Descr folder were already scaled during the descritpion_export_pipeline. 
    However, if you recalculate the RDKIT samples you should adjust to scaling
    them, too.
    """
    
    x_scaler = preprocessing.StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_scaler = preprocessing.StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)


    return X_train_scaled, y_train_scaled, y_scaler



class Model:
    def __init__(
            self,
            n_jobs: int = 30):
        """
        Initializes the Model class.
        
        Args:
            n_jobs (int): The number of jobs to run in parallel. Default is 30.
        
        Returns:
            None.
        """

        self.n_jobs = n_jobs
        self.influence_scores = None
        self.model = None
        self.fp_results = None
        self.fp_logs = None

    def calculate_influence(
            self, 
            dataset: HTSDataPreprocessor.DatasetContainer):
        """
        A placeholder method for calculating influence scores. This method is intended 
        to be implemented in subclasses.
        
        Args:
            dataset (HTSDataPreprocessor.DatasetContainer): The dataset for which influence scores are to be calculated.
        
        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """

        raise NotImplementedError("Subclasses should implement this method.")
        
    def log_results(
            self,
            dataset:HTSDataPreprocessor.DatasetContainer,
            influence_scores:np.ndarray,
            time_taken: float,
            replicate: int,
            results_box: np.ndarray,
            logs: pd.DataFrame
            ):
            """
            Logs the results of influence score calculations.
            
            Args:
                dataset (HTSDataPreprocessor.DatasetContainer): The dataset being processed.
                influence_scores (np.ndarray): The calculated influence scores for the dataset.
                time_taken (float): The time taken to calculate the influence scores.
                replicate (int): The replicate number of the calculation.
                results_box (np.ndarray): A container to store the results.
                logs (pd.DataFrame): A DataFrame to store logs.
            
            Returns:
                None. Updates results_box and logs with the calculated information.
            """

            r = replicate
            y_p_train, y_c_train, y_f_train, idx_train = get_labels(dataset.training_set)
            fp_rate_train = np.sum(y_f_train) / len(y_f_train)
            tp_rate_train = 1 - fp_rate_train
            mols_train = list(dataset.training_set["SMILES"])
            mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
            flags, flags_alt = process_ranking(y_p_train, influence_scores)    
            results_box[r,0] = time_taken
            results_box[r,1] = precision_score(y_f_train, flags_alt[idx_train])

            results_box[r,2] = precision_score(y_c_train, flags[idx_train])

            #get EF10 for FPs and TPs
            results_box[r,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
            results_box[r,4] = enrichment_factor_score(y_c_train, flags[idx_train])

            #get BEDROC20 for FPs and TPs
            results_box[r,5] = bedroc_score(y_f_train, influence_scores[idx_train], reverse=False)
            results_box[r,6] = bedroc_score(y_c_train, influence_scores[idx_train], reverse=True)

            #get diversity for FPs
            idx_fp = np.where(flags_alt == 1)[0]
            mols_fp = [mols_train[x] for x in idx_fp]
            results_box[r,7] = get_scaffold_rate(mols_fp)

            #get diversity for TPs
            idx_tp = np.where(flags == 1)[0]
            mols_tp = [mols_train[x] for x in idx_tp]
            results_box[r,8] = get_scaffold_rate(mols_tp)
            
            logs_temp = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                        flags[idx_train], influence_scores[idx_train], dataset.dataset_name)
            self.logs = pd.concat([logs, logs_temp], axis=0, ignore_index=True)
        
            
            return 
    
    def store_row(
            self,
            dataset,
            dataset_array: np.ndarray,
            replicates: int,
            index: int = 0
            ) -> pd.DataFrame:
        """Stores i-th dataset results in performance container for X datasets

        Args:
            analysis_array: (X,20) dataframe that stores results of a given
                            algorithm for all datasets
            dataset_array:  (1,9) array with the results of a given algorithm on
                            the i-th dataset
            replicates:     number of replicates to be saved
            fp_rate:        fraction of false positives in the confirmatory dataset
            tp_rate:        fraction of true positives in the confirmatory dataset
            index:          i-th row position to store results in

        Returns:
            Updated analysis array with results stored in the correct row (not the
            most elegant solution but at least it provides a straightforward way
            to handle both single and multi dataset performance collection)
        """
        y_p_train, y_c_train, y_f_train, idx_train = get_labels(dataset.training_set)
        fp_rate = np.sum(y_f_train) / len(y_f_train)
        tp_rate = 1 - fp_rate
        column_number = 20 + 9 * replicates
        analysis_array = np.zeros((1, column_number))
        pos = 0
        analysis_array[index, pos] = np.mean(dataset_array[:,0])      #mean training time
        pos += 1
        analysis_array[index, pos] = np.std(dataset_array[:,0])       #STD training time
        pos +=1
        end_pos = pos + replicates
        analysis_array[index, pos:end_pos] = dataset_array[:,0]
        pos = end_pos

        analysis_array[index, pos] = fp_rate 
        pos +=1                         #baseline FP rate
        analysis_array[index, pos] = np.mean(dataset_array[:,1])      #mean precision@90 FP
        pos +=1
        analysis_array[index, pos] = np.std(dataset_array[:,1])       #STD precision@90 FP
        pos +=1
        end_pos = pos + replicates
        analysis_array[index, pos:end_pos] = dataset_array[:,1]
        pos = end_pos

        analysis_array[index, pos] = tp_rate                          #baseline TP rate
        pos +=1
        analysis_array[index, pos] = np.mean(dataset_array[:,2])      #mean precision@90 TP
        pos +=1
        analysis_array[index, pos] = np.std(dataset_array[:,2])       #STD precision@90 TP
        pos +=1
        end_pos = pos + replicates
        analysis_array[index, pos:end_pos] = dataset_array[:,2]
        pos = end_pos

        analysis_array[index, pos] = np.mean(dataset_array[:,3])      #EF10 for FP
        pos +=1
        analysis_array[index, pos] = np.std(dataset_array[:,3])       #STD EF10 FP
        pos +=1
        end_pos = pos + replicates
        analysis_array[index, pos:end_pos] = dataset_array[:,3]
        pos = end_pos

        analysis_array[index, pos] = np.mean(dataset_array[:,4])     #EF10 for TP
        pos +=1
        analysis_array[index, pos] = np.std(dataset_array[:,4])      #STD EF10 TP
        pos +=1
        end_pos = pos + replicates
        analysis_array[index, pos:end_pos] = dataset_array[:,4]
        pos = end_pos

        analysis_array[index, pos] = np.mean(dataset_array[:,5])     #BEDROC20 for FP
        pos +=1
        analysis_array[index, pos] = np.std(dataset_array[:,5])      #STD
        pos +=1
        end_pos = pos + replicates
        analysis_array[index, pos:end_pos] = dataset_array[:,5]
        pos = end_pos

        analysis_array[index, pos] = np.mean(dataset_array[:,6])     #BEDROC20 for TP
        pos +=1
        analysis_array[index, pos] = np.std(dataset_array[:,6])      #STD
        pos +=1
        end_pos = pos + replicates
        analysis_array[index, pos:end_pos] = dataset_array[:,6]
        pos = end_pos

        analysis_array[index, pos] = np.mean(dataset_array[:,7])     #means FP scaffold diversity
        pos +=1
        analysis_array[index, pos] = np.std(dataset_array[:,7])      #STD FP scaffold diversity
        pos +=1
        end_pos = pos + replicates
        analysis_array[index, pos:end_pos] = dataset_array[:,7]
        pos = end_pos

        analysis_array[index, pos] = np.mean(dataset_array[:,8])     #means TP scaffold diversity
        pos +=1
        analysis_array[index, pos] = np.std(dataset_array[:,8])      #STD TP scaffold diversity
        pos +=1
        analysis_array[index, pos:] = dataset_array[:,8]
        
        # Add column names for replicates
        column_names = [
            "Time - mean", "Time - STD"]
        for j in range(replicates): # Each of the five replicates
            column_names.append("Time - Replicate {0}".format(j+1))
        column_names.append("FP rate")
        column_names.append("FP Precision@90 - mean")
        column_names.append("FP Precision@90 - STD")
        for j in range(replicates): # Each of the five replicates
            column_names.append("FP Precision@90 - Replicate {0}".format(j+1))
        column_names.append("TP rate")
        column_names.append("TP Precision@90 - mean")
        column_names.append("TP Precision@90 - STD")
        for j in range(replicates): # Each of the five replicates
            column_names.append("TP Precision@90 - Replicate {0}".format(j+1))
        column_names.append("FP EF10 - mean")
        column_names.append("FP EF10 - STD")
        for j in range(replicates): # Each of the five replicates
            column_names.append("FP EF10 - Replicate {0}".format(j+1))
        column_names.append("TP EF10 - mean")
        column_names.append("TP EF10 - STD")
        for j in range(replicates): # Each of the five replicates
            column_names.append("TP EF10 - Replicate {0}".format(j+1))
        column_names.append("FP BEDROC20 - mean")
        column_names.append("FP BEDROC20 - STD")
        for j in range(replicates): # Each of the five replicates
            column_names.append("FP BEDROC20 - Replicate {0}".format(j+1))
        column_names.append("TP BEDROC20 - mean")
        column_names.append("TP BEDROC20 - STD")
        for j in range(replicates): # Each of the five replicates
            column_names.append("TP BEDROC20 - Replicate {0}".format(j+1))
        column_names.append("FP Scaffold- mean")
        column_names.append("FP Scaffold - STD")
        for j in range(replicates): # Each of the five replicates
            column_names.append("FP Scaffold - Replicate {0}".format(j+1))
        column_names.append("TP Scaffold- mean")
        column_names.append("TP Scaffold - STD")
        for j in range(replicates): # Each of the five replicates
            column_names.append("TP Scaffold - Replicate {0}".format(j+1))
            
      
        
        db = pd.DataFrame(
                    data = analysis_array,
                    index = [dataset.dataset_name],
                    columns = column_names
                    )
        
        return db

    def apply_false_positive_identification(
            self, 
            dataset: HTSDataPreprocessor.DatasetContainer, 
            replicates: int = 1
            ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Applies the false positive identification process on a dataset.
    
        Args:
            dataset (HTSDataPreprocessor.DatasetContainer): The dataset on which to apply the process.
            replicates (int): The number of replicates for the process. Default is 1.
        
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: A tuple containing the results array and the logs DataFrame.
        """

        self.results_box = np.zeros((replicates,9))
        self.logs = pd.DataFrame([])
        for r in range(replicates):
            start = time.time()
            influence_scores = self.calculate_influence(dataset, seed = r)
            end = time.time() - start
            self.log_results(dataset = dataset, influence_scores = influence_scores, time_taken = end, replicate = r, results_box = self.results_box, logs=self.logs)
        self.results = self.store_row(dataset, self.results_box, replicates)
        return self.results, self.logs 

    def apply_active_learning(
            self,
            dataset: HTSDataPreprocessor.DatasetContainer,
            step_size: int = 1,
            steps: int = 6,
            regression_function: str = "lgbm",
            sampling_function: str = "greedy"
            ) -> List[int]:
        """
        Applies active learning to a dataset by iteratively screening small batches of compounds, calculating their influence scores, and 
        predicting the most influential samples of the remaining dataset for subsequent batch sampling. Each batch the number of actives 
        found is counted and returned as the results
        
        Args:
            dataset (HTSDataPreprocessor.DatasetContainer): The dataset to apply active learning.
            step_size (int): The step size for active learning. Default is 1.
            steps (int): The number of steps to perform in active learning. Default is 6.
            regression_function (str): The regression function to use. Default is "lgbm".
            sampling_function (str): The sampling function to use. Default is "greedy".
        
        Returns:
            List[int]: A list of the number of actives found at each step.
        """

        #initialize result list
        actives_found = []
        top_idx = []
        
        #create new dataset instances for active learning steps
        sampled_dataset = HTSDataPreprocessor.DatasetContainer(dataset.dataset_name)
        rest_dataset = HTSDataPreprocessor.DatasetContainer(dataset.dataset_name)
        
        idx_rest, idx_sampled, _, _ = train_test_split(range(len(dataset.training_set_labels)), dataset.training_set_labels, test_size=0.01*step_size, random_state=0)
        
        sampled_dataset.training_set_descriptors = dataset.training_set_descriptors[idx_sampled]
        sampled_dataset.training_set = dataset.training_set.loc[idx_sampled]
        sampled_dataset.validation_set_descriptors = dataset.validation_set_descriptors
        sampled_dataset.validation_set = dataset.validation_set
        
        rest_dataset.training_set_descriptors = dataset.training_set_descriptors[idx_rest]
        rest_dataset.training_set = dataset.training_set.loc[idx_rest]
        rest_dataset.validation_set_descriptors = dataset.validation_set_descriptors
        rest_dataset.validation_set = dataset.validation_set                                          
        sampled_dataset.training_set_labels = sampled_dataset.training_set["Primary"]
        sampled_dataset.validation_set_labels = sampled_dataset.validation_set["Confirmatory"]
        rest_dataset.training_set_labels = rest_dataset.training_set["Primary"]   
        rest_dataset.validation_set_labels = rest_dataset.validation_set["Confirmatory"] 
        for i in range(steps):
            #calculate influence scores on current batch
            sampled_influence_scores = self.calculate_influence(sampled_dataset)
            sampled_influence_scores = sampled_influence_scores.reshape(-1,1)
            _, sampled_influence_scores, y_scaler = transform_data(sampled_dataset.training_set_descriptors, sampled_influence_scores)
            
            if(regression_function.lower()=="lgbm"):
                #if lgbm is the regression function, the influence scores have to be in a special format
                sampled_influence_scores = np.ravel(sampled_influence_scores)
            #predict influence scores on remaining dataset using the regresssion function of choice
            predictions_mean, predictions_var = rf.regression_functions_dict[regression_function.lower()](sampled_dataset.training_set_descriptors, sampled_influence_scores, rest_dataset.training_set_descriptors, _, 0)
           #sample the next batch according to the sampling function of choice given the prediction of the regression model
            if(sampling_function == "greedy"):
                predictions = sf.greedy(predictions_mean)
                idx_rest_sorted = np.argsort(predictions, axis= 0)
            elif(sampling_function == "ucb"):
                predictions = sf.ucb(predictions_mean, predictions_var, 2)
                idx_rest_sorted = np.argsort(predictions, axis= 0) 
                
            #add the new samples to the batch of screened samples and remove them from the remaining dataset
            top_idx.append(idx_rest_sorted[:min(len(rest_dataset.training_set_labels), int(len(dataset.training_set_labels) / (100/step_size)))].ravel().tolist())
            if i == 0:
                actives_found.append(np.sum(sampled_dataset.training_set_labels))
            sampled_dataset.training_set_descriptors = np.concatenate([sampled_dataset.training_set_descriptors, rest_dataset.training_set_descriptors[top_idx[i]]])
            rest_dataset.training_set_descriptors = np.delete(rest_dataset.training_set_descriptors, top_idx[i], 0)
            
            new_batch =  rest_dataset.training_set.iloc[top_idx[i]]
            sampled_dataset.training_set = pd.concat([sampled_dataset.training_set, new_batch], ignore_index=True)
            rest_dataset.training_set = rest_dataset.training_set.drop(rest_dataset.training_set.index[top_idx[i]])
            sampled_dataset.training_set_labels = sampled_dataset.training_set["Primary"]
            rest_dataset.training_set_labels = rest_dataset.training_set["Primary"]                                          
            #record the number of active samples in the screened batch
            actives_found.append(np.sum(sampled_dataset.training_set_labels))
        
        return actives_found
    
    def greedy_active_learning(
            self,
            dataset: HTSDataPreprocessor.DatasetContainer,
            step_size: int = 1,
            steps: int = 6,
            model: Optional = None
            ) -> List[int]:
        """
          Executes an active learning process on a given dataset, progressively selecting and training on small subsets of data to identify 'active' compounds. 
          It involves iteratively training a model (default or specified) on a sampled dataset and using it to predict the most promising samples in the remaining 
          dataset for subsequent training. This approach aims to efficiently find active compounds with minimal data.
        
          Args:
              dataset (HTSDataPreprocessor.DatasetContainer): The dataset container with training and validation sets.
              step_size (int): Determines the percentage of the dataset to sample at each step. Default is 1.
              steps (int): The total number of iterative steps to perform in the active learning process. Default is 6.
              model (Optional[MachineLearningModel]): The machine learning model to use. If None, a default model is used.
        
          Returns:
              List[int]: A list containing the count of active compounds found in each step of the active learning process.
          """
        #initialize result list
        actives_found = []
        top_idx = []
        
        #create new dataset instances for active learning steps
        sampled_dataset = HTSDataPreprocessor.DatasetContainer(dataset.dataset_name)
        rest_dataset = HTSDataPreprocessor.DatasetContainer(dataset.dataset_name)
        
        idx_rest, idx_sampled, _, _ = train_test_split(range(len(dataset.training_set)), dataset.training_set, test_size=0.01*step_size, random_state=0) 
        
        sampled_dataset.training_set_descriptors = dataset.training_set_descriptors[idx_sampled]
        sampled_dataset.training_set = dataset.training_set.loc[idx_sampled]
        sampled_dataset.validation_set_descriptors = dataset.validation_set_descriptors
        sampled_dataset.validation_set = dataset.validation_set
        
        rest_dataset.training_set_descriptors = dataset.training_set_descriptors[idx_rest]
        rest_dataset.training_set = dataset.training_set.loc[idx_rest]
        rest_dataset.validation_set_descriptors = dataset.validation_set_descriptors
        rest_dataset.validation_set = dataset.validation_set                                               
        sampled_dataset.training_set_labels = sampled_dataset.training_set["Primary"]
        sampled_dataset.validation_set_labels = sampled_dataset.validation_set["Confirmatory"]
        rest_dataset.training_set_labels = rest_dataset.training_set["Primary"]   
        rest_dataset.validation_set_labels = rest_dataset.validation_set["Confirmatory"] 
        for i in range(steps):
            start = time.time()
            if model == None:
                    model = lightgbm.LGBMClassifier(n_jobs = self.n_jobs)
            model.fit(sampled_dataset.training_set_descriptors, sampled_dataset.training_set_labels)
            prediction_score = model.predict_proba(rest_dataset.training_set_descriptors)[:,0]
            idx_rest_sorted = np.argsort(prediction_score)
            top_idx.append(idx_rest_sorted[:min(len(rest_dataset.training_set_labels), int(len(dataset.training_set_labels) / (100/step_size)))].tolist())
            if i == 0:
                actives_found.append(np.sum(sampled_dataset.training_set_labels))
            sampled_dataset.training_set_descriptors = np.concatenate([sampled_dataset.training_set_descriptors, rest_dataset.training_set_descriptors[top_idx[i]]])
            rest_dataset.training_set_descriptors = np.delete(rest_dataset.training_set_descriptors, top_idx[i], 0)
            
            new_batch =  rest_dataset.training_set.iloc[top_idx[i]]
            sampled_dataset.training_set = pd.concat([sampled_dataset.training_set, new_batch], ignore_index=True)
            rest_dataset.training_set = rest_dataset.training_set.drop(rest_dataset.training_set.index[top_idx[i]])
            sampled_dataset.training_set_labels = sampled_dataset.training_set["Primary"]
            rest_dataset.training_set_labels = rest_dataset.training_set["Primary"]    
            actives_found.append(np.sum(sampled_dataset.training_set_labels))
        
        return actives_found
            
    def random_active_learning(
            self, 
            dataset: HTSDataPreprocessor.DatasetContainer,
            step_size: int = 1,
            steps: int = 6
            ) -> List[int]:
        """
        Implements a random active learning strategy on a dataset. This method involves randomly selecting small subsets 
        of data across multiple iterations to identify active compounds. Unlike structured active learning methods, 
        this approach does not rely on model predictions to choose the next batch of data. Instead, it randomly samples 
        from the remaining dataset at each step, providing a baseline for comparison with more structured active learning strategies.
    
        Args:
            dataset (HTSDataPreprocessor.DatasetContainer): The dataset container with training and validation sets.
            step_size (int): Specifies the percentage of the dataset to sample at each step. Default is 1.
            steps (int): The total number of iterations to perform in the active learning process. Default is 6.
    
        Returns:
            List[int]: A list containing the number of active compounds identified in each iteration of the process.
        """
        #initialize result list
        actives_found = []
        top_idx = []
        
        #create new dataset instances for active learning steps
        sampled_dataset = HTSDataPreprocessor.DatasetContainer(dataset.dataset_name)
        rest_dataset = HTSDataPreprocessor.DatasetContainer(dataset.dataset_name)
        
        idx_rest, idx_sampled, _, _ = train_test_split(range(len(dataset.training_set_labels)), dataset.training_set_labels, test_size=0.01*step_size, random_state=0) 
        
        sampled_dataset.training_set_descriptors = dataset.training_set_descriptors[idx_sampled]
        sampled_dataset.training_set = dataset.training_set.loc[idx_sampled]
        sampled_dataset.validation_set_descriptors = dataset.validation_set_descriptors
        sampled_dataset.validation_set = dataset.validation_set
        
        rest_dataset.training_set_descriptors = dataset.training_set_descriptors[idx_rest]
        rest_dataset.training_set = dataset.training_set.loc[idx_rest]
        rest_dataset.validation_set_descriptors = dataset.validation_set_descriptors
        rest_dataset.validation_set = dataset.validation_set     
        sampled_dataset.training_set_labels = sampled_dataset.training_set["Primary"]
        sampled_dataset.validation_set_labels = sampled_dataset.validation_set["Confirmatory"]
        rest_dataset.training_set_labels = rest_dataset.training_set["Primary"]   
        rest_dataset.validation_set_labels = rest_dataset.validation_set["Confirmatory"] 
        for i in range(steps):
            start = time.time()
            idx_rest_random = np.array(range(len(rest_dataset.training_set_labels)))
            random.shuffle(idx_rest_random)
            top_idx.append(idx_rest_random[:min(len(rest_dataset.training_set_labels), int(len(dataset.training_set_labels) / (100/step_size)))].tolist())
            if i == 0:
                actives_found.append(np.sum(sampled_dataset.training_set_labels))
            sampled_dataset.training_set_descriptors = np.concatenate([sampled_dataset.training_set_descriptors, rest_dataset.training_set_descriptors[top_idx[i]]])
            rest_dataset.training_set_descriptors = np.delete(rest_dataset.training_set_descriptors, top_idx[i], 0)
            
            new_batch =  rest_dataset.training_set.iloc[top_idx[i]]
            sampled_dataset.training_set = pd.concat([sampled_dataset.training_set, new_batch], ignore_index=True)
            rest_dataset.training_set = rest_dataset.training_set.drop(rest_dataset.training_set.index[top_idx[i]])
            sampled_dataset.training_set_labels = sampled_dataset.training_set["Primary"]
            rest_dataset.training_set_labels = rest_dataset.training_set["Primary"]    
            actives_found.append(np.sum(sampled_dataset.training_set_labels))
        
        return actives_found
        
    def apply_undersampling(
            self,
            dataset:HTSDataPreprocessor.DatasetContainer,
            model: Optional = None,
            steps: int = 19
            ) -> pd.DataFrame:
        """
          Applies undersampling techniques to a dataset and evaluates the performance of a machine learning model 
          (either provided or a default one) on the undersampled datasets. The function systematically removes a 
          certain percentage of the samples (increasing in steps) from the dataset and assesses the model's performance on these subsets. 
          It conducts this process for both low and high influence undersampled datasets, providing insights into how different 
          undersampling strategies affect model performance.
        
          Args:
              dataset (HTSDataPreprocessor.DatasetContainer): The dataset container with training and validation sets.
              model (Optional): The machine learning model to be used. If None, a default model is used.
        
          Returns:
              pd.DataFrame: A DataFrame containing the removal counts and the corresponding performance metrics (PRAUC) for the basic, 
              high influence, and low influence models for each undersampling level.
         """
    
        results_list = []
        remove_count = []        
        for rm in range(steps):
            rm = rm+1
            remove_count.append(int(len(dataset.training_set_labels)*0.05 * rm))
        
        for i in remove_count:
            if (i > len(dataset.training_set_labels)):
                break 
            low_influence_undersampled_dataset = self.undersample(dataset, num_neg_removable = i)
            high_influence_undersampled_dataset = self.undersample(dataset, num_neg_removable = i, remove_high = True)
                                                                    
            
            #basic model
            num_pos = dataset.training_set_labels.sum()
            num_neg = len(dataset.training_set_labels) - num_pos
            weights_pos_sqrt = np.sqrt(1 / (num_pos / (num_neg + num_pos)))
            weights_neg_sqrt = np.sqrt(1 / (num_neg / (num_neg + num_pos)))
            class_weight_basic = {0: weights_neg_sqrt, 1: weights_pos_sqrt}
            basic_model = model
            if model == None:
                basic_model = lightgbm.LGBMClassifier(class_weight=class_weight_basic, n_jobs = self.n_jobs)
            basic_model.fit(dataset.training_set_descriptors,dataset.training_set_labels)
            pred_basic_val = basic_model.predict_proba(dataset.validation_set_descriptors)[:,1]
            
            #high_model
            high_model = model
            if model == None:
                high_model = lightgbm.LGBMClassifier(class_weight=class_weight_basic, n_jobs = self.n_jobs)
            high_model.fit(high_influence_undersampled_dataset.training_set_descriptors,
                            high_influence_undersampled_dataset.training_set_labels)
            pred_high_val = high_model.predict_proba(high_influence_undersampled_dataset.validation_set_descriptors)[:,1]
            
            #low_model
            low_model = model
            if model == None:
                low_model = lightgbm.LGBMClassifier(class_weight=class_weight_basic, n_jobs = self.n_jobs)
            low_model.fit(low_influence_undersampled_dataset.training_set_descriptors,
                            low_influence_undersampled_dataset.training_set_labels)
            pred_low_val = low_model.predict_proba(low_influence_undersampled_dataset.validation_set_descriptors)[:,1]
            
             
            prauc_basic_val = average_precision_score(dataset.validation_set_labels, pred_basic_val)
            print("Basic model Validation Performance: ", prauc_basic_val)
    
            prauc_high_val = average_precision_score(dataset.validation_set_labels, pred_high_val)
            print("High model Validation Performance: ", prauc_high_val)
    
            prauc_low_val = average_precision_score(dataset.validation_set_labels, pred_low_val)
            print("Low model Validation Performance: ", prauc_low_val)
    
            results_list.append((i, (prauc_basic_val,prauc_high_val,prauc_low_val )))
        
        percentages = []
        for rm in range(steps):
            rm = rm+1
            percentages.append(round(0.05 * rm, 3))
            
        #translate the results into a list of dataframes
        dataframe_list = []
        for i in [results_list]:
            index = percentages
            counts = [x[0] for x in i]
            basic = [x[1][0] for x in i]
            high = [x[1][1] for x in i]
            low = [x[1][2] for x in i]
            df = pd.DataFrame(index = index, data = {"counts": counts, "basic": basic, "high": high, "low": low})
        return df                                       
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
    def undersample( self,
        dataset: HTSDataPreprocessor.DatasetContainer,
        ratio: Optional[float] = None,
        num_neg_removable: Optional[int] = None,
        remove_high: bool = False,
        random: bool = False
    ) -> HTSDataPreprocessor.DatasetContainer:
        """
        Performs undersampling on a given dataset. The function removes a specified number of negative samples (inactive compounds) to balance
        the dataset or achieve a desired positive-to-negative ratio. It can remove samples with the highest influence scores, the lowest, or randomly, 
        based on the provided parameters. This method is useful for handling class imbalance in datasets.
    
        Args:
            dataset (HTSDataPreprocessor.DatasetContainer): The dataset container with training set labels and descriptors.
            ratio (Optional[float]): The target ratio of positive to negative samples. If specified, 'num_neg_removable' is calculated accordingly.
            num_neg_removable (Optional[int]): The specific number of negative samples to remove. Used if 'ratio' is not specified.
            remove_high (bool): If True, removes samples with the highest influence scores. Defaults to False, which removes samples with the lowest scores.
            random (bool): If True, removes samples randomly. Overrides 'remove_high' if True.
    
        Returns:
            HTSDataPreprocessor.DatasetContainer: The undersampled dataset.
        """
        influence_scores = self.calculate_influence(dataset)
        undersampled_dataset = copy.deepcopy(dataset)
        num_pos = np.sum(dataset.training_set_labels)
        num_neg = len(dataset.training_set_labels) - num_pos
        
        if ratio != None:
            num_neg_target = int(num_pos/ratio)
            num_neg_removable = num_neg - num_neg_target
        
        if random:
            indices = np.arange(len(dataset.training_set_labels))
            random.shuffle(indices)
        else: 
            indices = np.argsort(influence_scores, axis=0)

        indices_inactives = []
        
        for i in range(len(indices)):
            if dataset.training_set_labels[indices[i]] == 0:
                indices_inactives.append(indices[i])
        indices = np.array(indices_inactives)
        if remove_high:
            undersampled_dataset = copy.deepcopy(dataset)
            undersampled_dataset.training_set_descriptors = np.delete(undersampled_dataset.training_set_descriptors, indices[-num_neg_removable:], axis=0)
            undersampled_dataset.training_set = undersampled_dataset.training_set.drop(indices[-num_neg_removable:])
            undersampled_dataset.training_set = undersampled_dataset.training_set.reset_index(drop=True)
            undersampled_dataset.training_set_labels = undersampled_dataset.training_set_labels.drop(indices[-num_neg_removable:])
            undersampled_dataset.training_set_labels = undersampled_dataset.training_set_labels.reset_index(drop=True)
        else:
            undersampled_dataset = copy.deepcopy(dataset)
            undersampled_dataset.training_set_descriptors = np.delete(undersampled_dataset.training_set_descriptors, indices[:num_neg_removable], axis=0)
            undersampled_dataset.training_set = undersampled_dataset.training_set.drop(indices[:num_neg_removable])
            undersampled_dataset.training_set = undersampled_dataset.training_set.reset_index(drop=True)
            undersampled_dataset.training_set_labels = undersampled_dataset.training_set_labels.drop(indices[:num_neg_removable])
            undersampled_dataset.training_set_labels = undersampled_dataset.training_set_labels.reset_index(drop=True)
        
        return undersampled_dataset
        
 
     
    
