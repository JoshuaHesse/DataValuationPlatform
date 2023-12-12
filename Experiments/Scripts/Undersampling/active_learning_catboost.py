#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:02:31 2023

@author: joshua

This file contains the function used for active learning via catboost, using
self importance (training samples importances on their own prediction), or 
test importance (training samples influences on validation set prediction)
"""

from catboost import Pool
from catboost import CatBoostClassifier as clf
import numpy as np
import time

def influences_catboost_self(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray = None,
        y_val: np.ndarray = None,
        i: int = 0,
        seed : int = 0
        )-> np.ndarray:
    """
    

    Parameters
    ----------
  x_train : np.ndarray (M, D)
      Vectorized molecular representations of M molecules, D dimensions 
      per molecule
  y_train : np.ndarray (M, 1)
      Labels for M molecules, binary
  x_val : np.ndarray (N, D)
      Vectorized molecular representations of N validation molecules, D dimensions 
      per molecule. The default is none as this function only uses self-influence.
  y_val : np.ndarray (N, 1)
      Labels for N validation molecules, binary.
      The default is none as this function only uses self-influence.
  i : int
      ith step of iteration. The default is 0.
  seed: int, optional
      seed to set random seed of mvsa. The default is 0.
    

    Returns
    -------
    influences: np.ndarray (M, 1)
        Influences for all M molecules calculated via Catboost
    
    ------
    
    Function trains a Catboost model and predicts influence scores of all samples
    on their own prediction using catboosts built-in object-importance functions    

    """
    start = time.time()
    #catboost model
    model = clf(iterations=100, random_state = seed)
    model.fit(x_train, y_train, verbose=False)
    #both pools contain all samples. This will result in the influence score
    #representing the influence of every sample on all other samples
    pool1 = Pool(x_train, y_train)
    pool2 = Pool(x_train, y_train)
    #influence calculation, using "AllPoints" as it is more accurate than
    #"SinglePoint" - however it also takes longer -- when the training pool gets
    #too big, maybe change update_method to the less accurate but faster version
    idx_cat_self, scores_self = model.get_object_importance(pool =pool2, 
                                                            train_pool =pool1,
                                                            update_method="AllPoints",
                                                            importance_values_sign="All")
    vals_self = np.empty((y_train.shape[0]))
    vals_self[idx_cat_self] = scores_self
    end = time.time() - start
    print("Catboost_self calculation time: " + str(end)+" in step " + str(i))
    influences = vals_self 
    return influences


def influences_catboost_test(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        i: int
        )-> np.ndarray:
    """
    

    Parameters
    ----------
  x_train : np.ndarray (M, D)
      Vectorized molecular representations of M molecules, D dimensions 
      per molecule
  y_train : np.ndarray (M, 1)
      Labels for M molecules, binary
  x_val : np.ndarray (N, D)
      Vectorized molecular representations of N validation molecules, D dimensions 
      per molecule
  y_val : np.ndarray (N, 1)
      Labels for N validation molecules, binary
  i : int
        ith step of iteration.
    

    Returns
    -------
    influences: np.ndarray (M, 1)
        Influences for all M molecules calculated via Catboost
    
    ------
    
    Function trains a Catboost model and predicts influence scores of all samples
    on the prediction of a test set using catboosts built-in object-importance functions.

    """
    start = time.time()
    #catboost model
    model = clf(iterations=100, random_state = 0)
    model.fit(x_train, y_train, verbose=False)
    #both pools contain all samples. This will result in the influence score
    #representing the influence of every sample on all other samples
    pool1 = Pool(x_train, y_train)
    pool2 = Pool(x_val, y_val)
    #influence calculation, using "AllPoints" as it is more accurate than
    #"SinglePoint" - however it also takes longer -- when the training pool gets
    #too big, maybe change update_method to the less accurate but faster version
    idx_cat_self, scores_self = model.get_object_importance(pool = pool2, 
                                                            train_pool = pool1,
                                                            update_method="AllPoints",
                                                            importance_values_sign="All")
    #create result array in the shape of the training labels
    vals_self = np.empty((y_train.shape[0]))
    #assign the scores to the correct indices 
    vals_self[idx_cat_self] = scores_self
    end = time.time() - start
    print("Catboost_test calculation time: " + str(end)+" in step " + str(i))
    influences = vals_self 
    return influences
