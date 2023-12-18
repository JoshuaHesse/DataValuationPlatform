#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:11:48 2023

@author: joshua

This file contains the function used to calculate training sample influence
scores on the prediction of a validation set using Data Valuation via Reinforcement
Learning (DVRL) 
"""
import warnings
import numpy as np
import time
#tensorflow and dvrl arent present in the datascope environment,
#and would throw errors otherwhise
try:    
    from dvrl import Dvrl
    import tensorflow as tf
except ImportError as e:
    warnings.warn('DVRL failed to import', ImportWarning)
import lightgbm


#-----------------------------------------------------------------------------#
def influences_dvrl(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        i: int
        )-> np.ndarray:
    # Network parameters for the DVRL model
    # Network parameters were slightly modified from standard:
    
    parameters = dict()
    parameters['hidden_dim'] = 100
    parameters['comb_dim'] = 10
    #standard is 2000 iterations -> down to 1000 to lower time
    parameters['iterations'] = 1000
    parameters['activation'] = tf.nn.relu
    parameters['layer_number'] = 5
    #standard is 2000 for batch size, increased to 5000 as otherwhise sometimes
    #there is only one class in the batch - causing DVRL to break
    parameters['batch_size'] = 5000
    parameters['learning_rate'] = 0.01
    
 
    #set timer to determine time per iteration
    start = time.time()
    
    # Sets checkpoint file name
    checkpoint_file_name = './tmp/active_learning_dvrl_model.ckpt'

    # Defines predictive model and problem type
    pred_model = lightgbm.LGBMClassifier(n_jobs=50, random_state=0)
    #pred_model = linear_model.LogisticRegression(max_iter=500, solver='lbfgs')
    problem = 'classification'

    # Flags for using stochastic gradient descent / pre-trained model
    flags = {'sgd': False, 'pretrain': False}

    # Initalizes DVRL: x_val and y_val are the confirmatory datapoints
    dvrl_class = Dvrl(x_train, y_train, x_val, y_val, 
                           problem, pred_model, parameters, checkpoint_file_name, flags)

    # Trains DVRL
    dvrl_class.train_dvrl('auc')
    end = time.time() - start
    print("DVRL calculation time: " + str(end)+" in step " + str(i))

    # Estimates data values
    influences = dvrl_class.data_valuator(x_train, y_train)
    return influences 
