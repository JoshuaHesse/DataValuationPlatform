#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:01:08 2023

@author: joshua

This file contains the function used for active learning via MVS_A
"""
from MVS_A import sample_analysis
import numpy as np
import time


def influences_mvsa(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray = None,
        y_val: np.ndarray = None,
        i: int = 0,
        seed: int = 0
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
    i: int, optional
        ith step of iteration. The default is 0.
    seed: int, optional
        seed to set random seed of mvsa. The default is 0.
    Returns
    -------
    influences: np.ndarray (M, 1)
        Influences for all M molecules calculated via MVS-A

    ------
    
    calculates influence scores of training samples via MVS-A, which is based
    on gradientboosted decision trees.
    """
    # mvs-a object creation via the mvsa class "sample_analysis" - for more detail,
    #read the MVS_A.py file
    obj = sample_analysis(x=x_train,
                          y=y_train,
                          params="default",
                          verbose=False,
                          seed=seed)
    start = time.time()
    
    #model creation
    obj.get_model()
    
    #influence prediction
    vals = obj.get_importance()
    end = time.time() - start
    print("MVSA calculation time: " + str(end)+" in step " + str(i))

    #as mvsa reports influences inversed to most other influence functions,
    # the results are reversed via *-1 to allow influencefunction-independent
    # workflows downstream
    influences = vals * -1

    return np.array(influences).reshape(-1, 1)
