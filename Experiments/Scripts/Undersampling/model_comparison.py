#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:49:46 2023

@author: joshua
"""

import numpy as np
from typing import Tuple
import random
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

def run_model_comparison(x_train: np.ndarray,
                         y_train: np.ndarray,
                         x_test: np.ndarray, 
                         y_test: np.ndarray,
                         x_val: np.ndarray,
                         y_val: np.ndarray,
                         influences: np.ndarray,
                         influence_method: str,
                         num_removed: int,
                         replicate: int,
                         inactives_only: bool = True,
                         class_weights: bool = False
                         )-> Tuple[float,float,float]:
    """
    

    Parameters
    ----------
    x_train : np.ndarray (M,D)
        DESCRIPTION: Features of M training samples with D features 
    y_train : np.ndarray (M,)
        DESCRIPTION: Labels of M training samples (either 1 or 0)
    x_test : np.ndarray (N,D)
        DESCRIPTION: Features of N test samples with D features 
                    These are a subset of the validation set, representing 
                    the same chemical domain. These samples are the subset 
                    with which the model will be adapted to the new domain
    y_test : np.ndarray (N,)
        DESCRIPTION: Labels of N test samples (either 1 or 0)
    x_val : np.ndarray (V,D)
        DESCRIPTION: Features of V validation samples with D features 
                    These are a structurally different from the training set,
                    representing the domain the model should be adapted to
    y_val : np.ndarray (V,)
        DESCRIPTION: Labels of V validation samples (either 1 or 0)
    influences : np.ndarray (M,)
        DESCRIPTION: influence scores for the M training samples
    Influence_method: str, 
        Description: Which influence method is used (MVSA, TracIn, Random)
    num_removed : int
        DESCRIPTION: number of samples that should be removed from the training
                    set upon adaptation
    inactives_only : Bool, optional
        DESCRIPTION: whether or not to allow active samples to be deleted during
                    adaptation. The default is True.
    class_weights : Bool, optional
        DESCRIPTION: whether or not to use class weights during LGBM training,
        the default is False.


    Returns
    -------
    prauc_basic_val : float
        DESCRIPTION: Average precision score for the basic model on
                    the validation set
    prauc_high_val : float
        DESCRIPTION: Average precision score for the high model on
                    the validation set (high model meaning the model where the
                    high influence score samples were removed)
    prauc_low_val : float
        DESCRIPTION: Average precision score for the low model on
                    the validation set (low model meaning where the low 
                                        influence score samples were removed)
                    
    _____________
    
    This function takes as input a training and validation set, as well as the 
    influence scores of said training set samples. Then, according to the 
    num_removed parameter, the function removes #num_removed samples according 
    to their influence score, either the most or least influential ones. 
    
    Then, 3 models are fitted and evaluated: 
        - the basic model, using the entire training set and class labels according
            to the inverse of the frequece ratios between the 2 classes
        
        - the high model, using the training set where the most influential 
            inactive samples were removed, class weights = 1
        
        - the low model, using the training set where the least influential 
            inactive samples were removed, class weights = 1
    
    The 3 models are compared in performance via the average precision score
    and the results are returned
        

    """
    indices_inactives = []
    #sort samples by influence score
    if influence_method.lower() != "random":
        indices = np.argsort(influences, axis=0)

    else:
        random.seed(replicate)
        indices = np.arange(len(y_train))
        random.shuffle(indices)
    #if inactives_only is true, take out all actives
    if inactives_only==True:
        for i in range(len(indices)):
            if y_train[indices[i]] == 0:
                indices_inactives.append(indices[i])
        #set indices to only contain indices of inactive samples
        indices = np.array(indices_inactives)
        
    #remove lowest influence samples from the training set
    x_train_low = np.delete(x_train, indices[:num_removed], axis=0)
    y_train_low = np.delete(y_train, indices[:num_removed], axis=0)
    print("Low samples removed, shape now: ", y_train_low.shape)
    print("Low samples removed, sum now: ", y_train_low.sum())
    
    #remove highest influence samples from the training set 
    x_train_high = np.delete(x_train, indices[-num_removed:], axis=0)
    y_train_high = np.delete(y_train, indices[-num_removed:], axis=0)
    print("High samples removed, shape now: ", y_train_high.shape)
    print("High samples removed, sum now: ", y_train_high.sum())
    print("basic samples shape now: ", y_train.shape)
    print("basic samples sum now: ", y_train.sum())
    
    #train the basic model
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    weights_pos_sqrt = 1
    
    
    #only calculate pos weights of there are more than 0, otherwhise 
    #division_by_zero error
    if num_pos > 0:
        weights_pos_sqrt = np.sqrt(1 / (num_pos / (num_neg + num_pos)))
    weights_neg_sqrt = np.sqrt(1 / (num_neg / (num_neg + num_pos)))
    class_weight_basic = {0: weights_neg_sqrt, 1: weights_pos_sqrt}
    #if classweights were set to False, set to None
    if (class_weights == False):
        class_weight_basic = None
        
    #if number of actives is 0, set to none, as otherwhise 
    #key error: 1
    if num_pos == 0:
        class_weight_basic = None
        
    #train basic model with the original training set 
    basic_model = LGBMClassifier(class_weight=class_weight_basic, n_jobs = 20)
    basic_model.fit(x_train,
                    y_train)
    
    #predict_proba instead of predict to be able to chose potentially better
    #threshild as LGBM always uses 0.5
    pred_basic_val = basic_model.predict_proba(x_val)[:,1]




    #same as for basic model, but using the adapted training set where 
    #highly influential samples were removed
    num_pos = y_train_high.sum()
    num_neg = len(y_train_high) - num_pos
    weights_pos_sqrt = 1
    
    if num_pos > 0:
        weights_pos_sqrt = np.sqrt(1 / (num_pos / (num_neg + num_pos)))
    weights_neg_sqrt = np.sqrt(1 / (num_neg / (num_neg + num_pos)))
    class_weight_high = class_weight_basic
    
    if (class_weights == False):
        class_weight_high = None
        
    if num_pos == 0:
        class_weight_high = None
    if num_neg == 0:
        class_weight_high = None
    high_model = LGBMClassifier(class_weight=class_weight_high, n_jobs = 20)
    high_model.fit(x_train_high,
                   y_train_high)
    
    pred_high_val = high_model.predict_proba(x_val)[:,1]
    
    
    
    

    #same as for basic model, but using the adapted training set where 
    #low influential samples were removed
    num_pos = y_train_low.sum()
    num_neg = len(y_train_low) - num_pos
    weights_pos_sqrt = 1
    if num_pos > 0:
        weights_pos_sqrt = np.sqrt(1 / (num_pos / (num_neg + num_pos)))
    weights_neg_sqrt = np.sqrt(1 / (num_neg / (num_neg + num_pos)))
    class_weight_low = class_weight_basic
    if (class_weights == False):
        class_weight_low = None
    if num_pos == 0:
        class_weight_low = None
    if num_neg == 0:
        class_weight_low = None
    low_model = LGBMClassifier(class_weight=class_weight_low, n_jobs = 20)
    low_model.fit(x_train_low,
                  y_train_low)
    pred_low_val = low_model.predict_proba(x_val)[:,1]

    
    prauc_basic_val = average_precision_score(y_val, pred_basic_val)
    print("Basic model Validation Performance: ", prauc_basic_val)
    
    prauc_high_val = average_precision_score(y_val, pred_high_val)
    print("High model Validation Performance: ", prauc_high_val)
    
    prauc_low_val = average_precision_score(y_val, pred_low_val)
    print("Low model Validation Performance: ", prauc_low_val)
    
    return prauc_basic_val, prauc_high_val, prauc_low_val