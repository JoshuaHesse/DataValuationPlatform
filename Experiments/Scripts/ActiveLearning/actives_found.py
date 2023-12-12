#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:28:38 2023

@author: joshua

This file contains the active learning functions organizing the combinations
of influence-, regression-, and sampling functions, as well as the greedy active
learning function and the random sampling function.
The file also contains a transformation function used inside the active learning 
functions.
"""
import lightgbm
import numpy as np
import random
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from influence_functions import influence_functions_dict    
from regression_functions import regression_functions_dict
import time
from sklearn import preprocessing
from sampling_functions import ucb, greedy

#-----------------------------------------------------------------------------#
def actives_found(x_train: np.ndarray,
                  y_train: np.ndarray,
                  x_val: np.ndarray,
                  y_val: np.ndarray,
                  representation: str = "ecfp", 
                  seed: int = 0,
                  steps: int = 5, 
                  step_size: float = 1,
                  random_size: float = 1,
                  influence_calc: str = "MVSA",
                  regression_calc: str = "LGBM", 
                  sampling_strategy: str = "greedy"
                  )-> List[np.ndarray]:
    """
    

    Parameters
    ----------
    x_train : np.ndarray (M, D)
        ECFP or RDKIt representations of the dataset samples.
    y_train : np.ndarray (M, 1)
        Activity labels of the dataset samples (1 or 0).
    x_val : np.ndarray (N, D)
        ECFP or RDKIt representations of the validation samples needed in
        some influence functions.
    y_val : np.ndarray (N, 1)
        Activity labels of the the validation samples needed in
        some influence functions.
    representation : str, optional
        Ehich representation is used. The default is "ecfp".
    seed : int, optional
        Seed used as random state during initial train test split. The default is 0
    steps : int, optional
        Number of steps that shoud be calculated. The default is 5.
    step_size : float, optional
        How big 1 step is. 1 means a step analyzes 1% of the dataset.
        The default is 1.
    random_size : float, optional
        How big the initial random subsample is. 1 means it contains 1% of the 
        Dataset randomly picked. The default is 1.
    influence_calc : str, optional
        Which influence function should be used. The default is "MVSA".
    regression_calc : str, optional
        Which regression function should be used. The default is "LGBM".
    sampling_strategy : str, optional
        Which sampling strategy should be used. The default is "greedy".

    Returns
    -------
    actives_found : List (steps+1)
        List containing the number of actives found within each step.
        The first value will be independent of the influence-, regression-, and
        sampling-strategy as it contains randomly chosen samples
        
    ------
    
    This function iterates over (step_size) percent of the given dataset (steps)
    times. Each time, it uses (influence_calc) to calculate the influence scores
    of the currently sampled sample-pool(in the first step only the randomly chosen
    samples). The calculated influences are then used to predict the influences 
    of the remaining "rest" samples via (regression_calc). From these predictions,
    a number of samples corresponding to (step_size) percent of the dataset 
    are chosen via (sampling_strategy). These samples are then added to the 
    current sampled pool. Then, the number of actives in the sampled pool is 
    counted and saved. This process is repeated (steps) times.

    """
    #initialize result list
    actives_found = []
    
    #top_idx will contain the indices of the samples to be picked in the next 
    #iteration
    top_idx = []

    #the initial split is random using the random seed,
    #with x_sampled having the size of 0.01 * random_size
    x_rest, x_sampled, y_rest, y_sampled = train_test_split(x_train, y_train, test_size=0.01*random_size, random_state=seed)


    #active learning is repeated steps times
    for i in range(steps):
        
        #first, the influence scores of the sampled set are calculated
        print("Influence calculation in run " + str(i) + " using " + influence_calc)
        
        #influences are calculated using the influence functino defined in influence calc
        influences = influence_functions_dict[influence_calc.lower()](x_sampled, y_sampled,x_val, y_val, i)
        
        #influences are scaled to allow more reproducible regression
        influences = influences.reshape(-1,1)
        _, influences, y_scaler = transform_data(x_sampled, influences)
        
        
        #Second, the sampled influence scores are used to predict the "rest" 
        print("Regression Model Prediction in run " + str(i) + " using " + regression_calc)
        
        #for lgbm, influence scores are raveled as lgbm throws a warning otherwhise
        if(regression_calc=="LGBM"):
            influences = np.ravel(influences)
            
        #regression is done with the regression function defined in regression_calc
        predictions_mean, predictions_var = regression_functions_dict[regression_calc.lower()](x_sampled, influences, x_rest, y_rest, i)
        
        
        #the predicted influence scores are used to chose new samples from "rest"
        #to add to the "sampled" pool
        print("Sampling next compounds in run "+ str(i) + " using " + sampling_strategy)
        if(sampling_strategy == "greedy"):
            predictions = greedy(predictions_mean)
            idx_rest_sorted = np.argsort(predictions, axis= 0)
        elif(sampling_strategy == "ucb"):
            predictions = ucb(predictions_mean, predictions_var, 2)
            idx_rest_sorted = np.argsort(predictions, axis= 0) 
        
        
        #the highest (step_size) % of the samples are picked
        top_idx.append(idx_rest_sorted[:min(len(y_rest), int(len(y_train) / (100/step_size)))].tolist())
        
        #if this is the first step, the actives in the initial train set are saved
        if i == 0:
            actives_found.append(np.sum(y_sampled))
        
        #the chosen "rest" samples are added to the "sampled" pool and deleted
        #from the "rest" pool
        #using the indexing via a list of indices adds a third dimension to
        #the resulting array which is subsequently deleted via .squeeze()
        #however, sometimes this doesnt happen so the presence of the third 
        #dimension is checked and if present it is deleted
        if(x_rest[top_idx[i]].ndim == 3):
            x_sampled = np.concatenate([x_sampled, x_rest[top_idx[i]].squeeze()])
            y_sampled = np.concatenate([y_sampled, y_rest[top_idx[i]].squeeze()])
            x_rest = np.delete(x_rest, top_idx[i], 0)
            y_rest = np.delete(y_rest, top_idx[i], 0)
        else:
            x_sampled = np.concatenate([x_sampled, x_rest[top_idx[i]]])
            y_sampled = np.concatenate([y_sampled, y_rest[top_idx[i]]])
            x_rest = np.delete(x_rest, top_idx[i], 0)
            y_rest = np.delete(y_rest, top_idx[i], 0)
        actives_found.append(np.sum(y_sampled))
        print("Number of actives found: " + str(actives_found[i]) + " in run " + str(i))
        print("-----------------------------------")
        print("\n")
    assert len(actives_found) == steps+1, "Wrong length of actives_found"
    return actives_found



#-----------------------------------------------------------------------------#
def greedy_actives_found(x_train: np.ndarray,
                         y_train: np.ndarray,
                         steps: int = 5, 
                         step_size: float = 1,
                         random_size: float = 1,
                         seed:int =0
                         )-> List[np.ndarray]:
    """
    

    Parameters
    ----------
    x_train : np.ndarray (M, D)
        ECFP or RDKIt representations of the dataset samples.
    y_train : np.ndarray (M, 1)
        Activity labels of the dataset samples (1 or 0).
    steps : int, optional
        Number of steps that shoud be calculated. The default is 5.
    step_size : float, optional
        How big 1 step is. 1 means a step analyzes 1% of the dataset.
        The default is 1.
    random_size : float, optional
        How big the initial random subsample is. 1 means it contains 1% of the 
        Dataset randomly picked. The default is 1.
    seed: int, optional
        Seed used as random_State during initial train_test_split. The Default
        is 0.
        
    Returns
    -------
    actives_found : List[np.ndarray] (steps+1)
        List containing the number of actives found within each step.
        The first value will be independent of the influence-, regression-, and
        sampling-strategy as it contains randomly chosen samples


    ------
    
    This function iterates over (step_size) percent of the given dataset (steps)
    times. Each time, it trains a LightGBMClassifier with the "sampled" pool to
    predict the activity of the "rest" samples. From these predictions,
    a number of samples corresponding to (step_size) percent of the dataset 
    are chosen. These samples are then added to the current sampled pool. 
    Then, the number of actives in the sampled pool is counted and saved. 
    This process is repeated (steps) times.
    
    The difference to the actives_found function is that here the activity is 
    predicted and these predictions are used for sampling, rather than appying 
    influence computations.
    """
        
    #initialize result array
    actives_found = []
    #initialize list that will contain the highest scoring samples' indices
    top_idx = []
    #the initial split is random, with x_sampled having the size of 0.01 * random_size
    x_rest, x_sampled, y_rest, y_sampled = train_test_split(x_train, y_train, test_size=0.01*random_size, random_state=seed)

    for i in range(steps):
        #a LGBM classifier is used for the greedy approach as it showed the best
        #performance during development
        print("Fitting LightGBM in run ", i)
        start = time.time()
        
        #the classifier is trained on the sampled set
        model = lightgbm.LGBMClassifier()
        model.fit(x_sampled, y_sampled)
        
        end = time.time() - start
        print("Fitting time: ", end)
        print("Predicting LightGBM in run ", i)
        start = time.time()
        #the probabilities are predicted for all rest samples
        score = model.predict_proba(x_rest)[:,0]
        end = time.time() - start
        print("Fitting time: ", end)
        #the scores are sorted and the highst scoring samples are picked for the
        #next iteration
        idx_rest_sorted = np.argsort(score)
        top_idx.append(idx_rest_sorted[:min(len(y_rest), int(len(y_train) / (100/step_size)))].tolist())

        #the chosen "rest" samples are added to the "sampled" pool and deleted
        #from the "rest" pool
        
        #using the indexing via a list of indices add a third dimension to
        #the resulting array which is subsequently deleted via .squeeze()
        #however, sometimes this doesnt happen so the presence of the third 
        #dimension is checked and if present it is deleted    
        if i == 0:
            actives_found.append(np.sum(y_sampled))
        if(x_rest[top_idx[i]].ndim == 3):
            x_sampled = np.concatenate([x_sampled, x_rest[top_idx[i]].squeeze()])
            y_sampled = np.concatenate([y_sampled, y_rest[top_idx[i]].squeeze()])
            x_rest = np.delete(x_rest, top_idx[i], 0)
            y_rest = np.delete(y_rest, top_idx[i], 0)
        else:
            x_sampled = np.concatenate([x_sampled, x_rest[top_idx[i]]])
            y_sampled = np.concatenate([y_sampled, y_rest[top_idx[i]]])
            x_rest = np.delete(x_rest, top_idx[i], 0)
            y_rest = np.delete(y_rest, top_idx[i], 0)
        print("Number of actives found: ", actives_found[i])
        actives_found.append(np.sum(y_sampled))
    return actives_found


def random_actives_found(x_train: np.ndarray,
                         y_train: np.ndarray,
                         steps: int = 5, 
                         step_size: float = 1,
                         random_size: float = 1,
                         seed:int =0
                         )-> List[np.ndarray]:
    """
    

    Parameters
    ----------
    x_train : np.ndarray (M, D)
        ECFP or RDKIt representations of the dataset samples.
    y_train : np.ndarray (M, 1)
        Activity labels of the dataset samples (1 or 0).
    steps : int, optional
        Number of steps that shoud be calculated. The default is 5.
    step_size : float, optional
        How big 1 step is. 1 means a step analyzes 1% of the dataset.
        The default is 1.
    random_size : float, optional
        How big the initial random subsample is. 1 means it contains 1% of the 
        Dataset randomly picked. The default is 1.
    seed: int, optional
        Seed used as random_state for initital train_test_split. The default is 0.
        
    Returns
    -------
    actives_found : List[np.ndarray] (steps+1, 1)
        List containing the number of actives found within each step.
        The first value will be independent of the influence-, regression-, and
        sampling-strategy as it contains randomly chosen samples


    ------
    
    This function iterates over (step_size) percent of the given dataset (steps)
    times. Each time, random samples are chosen from the dataset according to 
    step size, and added to the current sampled pool. 
    Then, the number of actives in the sampled pool is counted and saved. 
    This process is repeated (steps) times.
    
    The function simulates stepwhise analysis of a large dataset without 
    machine learning, leading to random sampling in every step.
    """
        
    #initialize result array
    actives_found = []
    #initialize list that will contain the highest scoring samples' indices
    top_idx = []
    #the initial split is random, with x_sampled having the size of 0.01 * random_size
    x_rest, x_sampled, y_rest, y_sampled = train_test_split(x_train, y_train, test_size=0.01*random_size, random_state=seed)

    for i in range(steps):


        #indices of the rest set are shuffled and then the sampled, to 
        #simulate random sampling 
        idx_rest_random = np.array(range(len(y_rest)))
        random.shuffle(idx_rest_random)
        top_idx.append(idx_rest_random[:min(len(y_rest), int(len(y_train) / (100/step_size)))].tolist())

        #the chosen "rest" samples are added to the "sampled" pool and deleted
        #from the "rest" pool
        
        #using the indexing via a list of indices add a third dimension to
        #the resulting array which is subsequently deleted via .squeeze()
        #however, sometimes this doesnt happen so the presence of the third 
        #dimension is checked and if present it is deleted    
        if i == 0:
            actives_found.append(np.sum(y_sampled))
        if(x_rest[top_idx[i]].ndim == 3):
            x_sampled = np.concatenate([x_sampled, x_rest[top_idx[i]].squeeze()])
            y_sampled = np.concatenate([y_sampled, y_rest[top_idx[i]].squeeze()])
            x_rest = np.delete(x_rest, top_idx[i], 0)
            y_rest = np.delete(y_rest, top_idx[i], 0)
        else:
            x_sampled = np.concatenate([x_sampled, x_rest[top_idx[i]]])
            y_sampled = np.concatenate([y_sampled, y_rest[top_idx[i]]])
            x_rest = np.delete(x_rest, top_idx[i], 0)
            y_rest = np.delete(y_rest, top_idx[i], 0)
        print("Number of actives found: ", actives_found[i])
        actives_found.append(np.sum(y_sampled))
    return actives_found


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
