#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:05:22 2023

@author: joshua

This file contains the function used for active learning via data shapley.
"""
import warnings
import numpy as np
import time 

#datascope can only be imported if the script is run in the datascope 
#environment, as datascope's numpy requirements clash with tensorflow's requirements
try:    
    from datascope.importance.common import SklearnModelAccuracy
    from datascope.importance.shapley import ShapleyImportance
except ImportError:
    warnings.warn('Data shapley failed to import', ImportWarning)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

def influences_shapley(
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
       Influences for all M molecules calculated via data shapley
       
   ------
   
   This function calculates influence scores using Shapley Importance.
   For FP detection, undersampling had been neccessary due to OOM issues.
   As this active learning training is only done on small batches, undersampling
   is avoidable. 
    """
    start = time.time()
    #shapley importance uses KNN to approximate the shapley values
    neigh = KNeighborsClassifier(n_neighbors=5)
    #as the datascope data shapley pipeline is based on using sklearn pipelines
    #as inputs, the model is loaded in a sklearn pipeline
    pipe_ecfp = Pipeline([('model', neigh)])
    
    #influences are computed using the shapley importance functionality from
    #datascope
    utility = SklearnModelAccuracy(pipe_ecfp)
    importance = ShapleyImportance(method="neighbor", utility=utility)
    influences = importance.fit(x_train, y_train).score(x_val, y_val)
    end = time.time()-start
    print("KNN calculation time: " + str(end)+" in step " + str(i))
    return influences

