#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:14:47 2023

@author: joshua

This file contains all regression functions used for active learning as well
as the regression_function_dict.
"""
import warnings
import numpy as np
from typing import Tuple, List
import time
import lightgbm
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor

#gpflow and tensorflow arent present in the datascope environment,
#and would throw errors otherwhise
try:
    from DataValuationPlatform.models.utils.custom_kernels import TanimotoSVR, Tanimoto, TanimotoSK
    import gpflow
    from gpflow.mean_functions import Constant
    from gpflow.utilities import print_summary
    from DataValuationPlatform.models.utils.regression_utils import build_and_compile_model
except ImportError:
    warnings.warn('GP_flow and co failed to import', ImportWarning)



def regression_lgbm(
        x_train: np.ndarray,
        influences: np.ndarray,
        x_rest:np.ndarray,
        y_rest: np.ndarray,
        i: int
        )-> Tuple[np.ndarray, np.ndarray]:
        """
        
    
        Parameters
        ----------
        x_train : np.ndarray (M, D)
            vectorized molecular representations of M molecules, D dimensions 
            per molecule of training set
        influences : np.ndarray (M, 1)
            Influences of each training molecule
        x_rest : np.ndarray (N, D)
            vectorized molecular representations of N molecules, D dimensions 
            per molecule that has to be predicted    
        y_rest : np.ndarray (N, )
            binary activity label for the rest_molecules
        i : int
            ith iteration of loop.
    
        Returns
        -------
        predictions_mean : np.ndarray (N, 1)
            Mean prediction for each of the N x_rest molecules
        predictions_var : np.ndarray (N, 1)
            prediction variance for each of the N x_rest molecules
        -------
        
        Function uses LightGBM, a gradientBoosting model, to predict 
        influence scores of a set of molecules by training the model with a set 
        of molecules and their influence scores.
    
        """
        #collect predictions over 5 iterations to allow mean and var calculations
        predictions_array = np.zeros((x_rest.shape[0], 5))
        for r in range(5):
            print("Fitting LightGBM in run " + str(i) + " and replicate " + str(r))
            start = time.time()
            #LGBM regressor for regression
            pred_model = lightgbm.LGBMRegressor(n_jobs = 30)
            #fit with training set influences
            pred_model.fit(x_train, influences)
            end = time.time() - start
            print("Fitting time: ", end)
            print("Predicting LightGBM in run ", i)
            start = time.time()
            #prediction of influence scores for x_rest
            predictions = pred_model.predict(x_rest)
            end = time.time() - start
            print("Fitting time: ", end)
            #collect replicates in the predictions_array
            predictions_array[:,r] = predictions
        
        #calculate mean and var for each molecule
        predictions_mean =  np.mean(predictions_array, axis=1)
        predictions_var =  np.var(predictions_array, axis=1)
        
        return predictions_mean, predictions_var

#-----------------------------------------------------------------------------#
def batch_data(features: np.ndarray,
               batch_size: int = 512
               )-> List[np.ndarray]:
    """
    

    Parameters
    ----------
    data : np.ndarray (M, D)
        Training set features of M molecules with D dimensions per molecule.
    batch_size : int, optional
        Number of samples per batch. The default is 512.

    Returns
    -------
    list
        List of length:(M/batch_size) of np.arrays, each of size (batch_size, D).
        
    -------
    
    This function batches a feature training set into a list of np.arrays

    """
    return [features[i:i + batch_size] for i in range(0, len(features), batch_size)]

#-----------------------------------------------------------------------------#  
def regression_svr(x_train: np.ndarray,
                   influences: np.ndarray,
                   x_rest: np.ndarray,
                   y_rest: np.ndarray,
                   i: int
                   )-> Tuple[np.ndarray, np.ndarray]:
    """
    
    
    Parameters
    ----------
    x_train : np.ndarray (M, D)
        vectorized molecular representations of M molecules, D dimensions 
        per molecule of training set
    influences : np.ndarray (M, 1)
        Influences of each training molecule
    x_rest : np.ndarray (N, D)
        vectorized molecular representations of N molecules, D dimensions 
        per molecule that has to be predicted    
    y_rest : np.ndarray (N, )
        binary activity label for the rest_molecules
    i : int
        ith iteration of loop.
    
    Returns
    -------
    predictions_mean : np.ndarray (N, 1)
        Mean prediction for each of the N x_rest molecules
    predictions_var : np.ndarray (N, 1)
        prediction variance for each of the N x_rest molecules
    -------
    
    This function uses support vector regressors to predict influence scores of
    a rest set by training on influence scores of a training set.
    """
    #features and labels need to be of same data type
    X_train = x_train.astype(np.float64)
    y_train = influences.astype(np.float64).flatten()
    print("Regression Model Fitting in run ", i)
    start = time.time()
   
    #save replicates in predictions_array
    predictions_array = np.zeros((x_rest.shape[0], 5))
    for r in range(5):
       #create sklearn SVR model with a custom Tanimoto kernel that
       #allows handling of high dimensional data
       regr = svm.SVR(kernel=TanimotoSVR())
       #fit with training set
       regr.fit(X_train, y_train)
       #batch the rest set, as predicting all at the same time caused 
       #memory issues
       x_rest_batched = batch_data(x_rest, 512)
       predictions = []
       for x in x_rest_batched:
           t = x.astype(np.float64)
           y_pred = regr.predict(t)
           predictions.append(y_pred)
       #save replicates in predictions array    
       predictions = np.concatenate(predictions)
       predictions_array[:,r] = predictions
           
        
    print("Regression Model Prediction in run ", i)



    end = time.time() - start
    print("Prediction time: ", end)
    
    predictions_mean =  np.mean(predictions_array, axis=1)
    predictions_var =  np.var(predictions_array, axis=1)
    return predictions_mean, predictions_var

#-----------------------------------------------------------------------------#
def regression_dnn(
        x_train: np.ndarray,
        influences: np.ndarray,
        x_rest:np.ndarray,
        y_rest: np.ndarray,
        i: int
        )-> Tuple[np.ndarray, np.ndarray]:
        """
            
        
        Parameters
        ----------
        x_train : np.ndarray (M, D)
            vectorized molecular representations of M molecules, D dimensions 
            per molecule of training set
        influences : np.ndarray (M, 1)
            Influences of each training molecule
        x_rest : np.ndarray (N, D)
            vectorized molecular representations of N molecules, D dimensions 
            per molecule that has to be predicted    
        y_rest : np.ndarray (N, )
            binary activity label for the rest_molecules
        i : int
            ith iteration of loop.
        
        Returns
        -------
        predictions_mean : np.ndarray (N, 1)
            Mean prediction for each of the N x_rest molecules
        predictions_var : np.ndarray (N, 1)
            prediction variance for each of the N x_rest molecules
        -------
        Function uses a deep neural network to calculate influence scores
        of a set of molecules by training the DNN with a set of molecules and
        their influence scores
        """
        
        #collect predictions over 5 iterations to allow mean and var calculations
        predictions_array = np.zeros((x_rest.shape[0], 5))
        for r in range(5):
            print("Fitting DNN in run " + str(i) + " and replicate " + str(r))
            start = time.time()
            #build regression DNN
            pred_model = build_and_compile_model(x_train.shape[-1])
            #fit model on influences (Epochs can be increased or decreased e.g. 
            #if performance is still increasing when 50 epochs are reached)
            pred_model.fit(x_train, influences, validation_split = 0.2, epochs = 50)
            end = time.time() - start
            print("Fitting time: ", end)
            print("Predicting DNN in run " + str(i) + " and replicate " + str(r))
            start = time.time()
            influences_rest = pred_model.predict(x_rest)
            end = time.time() - start
            print("Prediction time: ", end)

            predictions_array[:,r] = influences_rest.squeeze()
    
        predictions_mean =  np.mean(predictions_array, axis=1)
        predictions_var =  np.var(predictions_array, axis=1)
        return predictions_mean, predictions_var
    
#-----------------------------------------------------------------------------#  
    
def regression_gpr(x_train: np.ndarray,
                   influences: np.ndarray,
                   x_rest: np.ndarray,
                   y_rest: np.ndarray,
                   i: int
                   )-> Tuple[np.ndarray, np.ndarray]:
    """
    

    Parameters
    ----------
    x_train : np.ndarray (M, D)
        vectorized molecular representations of M molecules, D dimensions 
        per molecule of training set
    influences : np.ndarray (M, )
        Influences of each training molecule
    x_rest : np.ndarray (N, D)
        vectorized molecular representations of N molecules, D dimensions 
        per molecule that has to be predicted         
    y_rest : np.ndarray (N, )
        binary activity label for the rest_molecules
    i : int
        ith iteration in the loop.

    Returns
    -------
    predictions : np.ndarray (N, )
       Mean prediction for each of the N x_rest molecules.
    variances : np.ndarray (N, )
        Prediction variance for each of the N x_rest molecules.

    ------
    
    this function uses a Gaussian Process Regressor from gpflow to predict 
    the influence values of x_rest. It uses a custom kernel from 
    https://anonymous.4open.science/r/f160a0a2-0161-4d31-ba55-2e3aab2133d3/GP/kernels.py
    that was adapted to use high dimensional data.
    """
    #ensure input types are consistent in types, as GPR will throw errors otherwhise
    
    X_train = x_train.astype(np.float64)
    y_train = influences.astype(np.float64)
    
    print("Fitting GPR in run " + str(i))
    
    start = time.time()
    #set kernel
    kernel = Tanimoto()
    
    #as GPRs dont use sparce matrices, they cannot be trained with huge training
    #sets, esp. whith high dimensional data. Therefore the number of training 
    #samples is capped at 10500 (empirical value using ecfp). If there are more
    #than 10500 training samples, only the 10500 most influential samples are used
    if (len(y_train) > 10500):
        y_train_idx_sorted = np.argsort(y_train, axis = 0)
        y_train_idx_sorted = y_train_idx_sorted[:10500]
        y_train = y_train[y_train_idx_sorted].squeeze().reshape(-1,1)
        X_train = X_train[y_train_idx_sorted].squeeze()
        
    #build and optimize GPR
    model = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=kernel, noise_variance=1)
    opt = gpflow.optimizers.Scipy()
    


    opt.minimize(model.training_loss, model.trainable_variables)
    print_summary(model)
    end = time.time() - start
    print("Fitting time: ", end)


    print("Predicting GPR in run ", i)
    start = time.time()
    
    #as for the training set, also the samples used for inference cannot
    #exceed moderate numbers at a time. Therefore, the samples are batched
    #and predicted sequentially
    x_rest_batched = batch_data(x_rest, 512)
    predictions = []
    variances = []
    #actual_values = []
    for x in x_rest_batched:
        t = x.astype(np.float64)
        y_pred, y_var = model.predict_f(t)
        #y_pred = y_scaler.inverse_transform(y_pred) --> inverse transform not necessary?
        predictions.append(y_pred)
        variances.append(y_var.numpy())
    predictions = np.concatenate(predictions)
    variances = np.concatenate(variances)
    end = time.time() - start
    print("Prediction time: ", end)
    
    return predictions, variances


#-----------------------------------------------------------------------------# 



def regression_gpr_sklearn(x_train: np.ndarray,
                   influences: np.ndarray,
                   x_rest: np.ndarray,
                   y_rest: np.ndarray,
                   i: int
                   )-> Tuple[np.ndarray, np.ndarray]:
    """
    

    Parameters
    ----------
    x_train : np.ndarray (M, D)
        vectorized molecular representations of M molecules, D dimensions 
        per molecule of training set
    influences : np.ndarray (M, )
        Influences of each training molecule
    x_rest : np.ndarray (N, D)
        vectorized molecular representations of N molecules, D dimensions 
        per molecule that has to be predicted         
    y_rest : np.ndarray (N, )
        binary activity label for the rest_molecules
    i : int
        ith iteration in the loop.

    Returns
    -------
    predictions : np.ndarray (N, )
       Mean prediction for each of the N x_rest molecules.
    variances : np.ndarray (N, )
        Prediction variance for each of the N x_rest molecules.

    ------
    
    this function uses a Gaussian Process Regressor from sklearn to predict 
    the influence values of x_rest. Unlike the gpflow implementation, it does
    not use the GPU.
    """
    #ensure input types are consistent in types, as GPR will throw errors otherwhise
    X_train = x_train.astype(np.float64)
    y_train = influences.astype(np.float64)
# =============================================================================
#     if (len(y_train) > 30000):
#         y_train_idx_sorted = np.argsort(y_train, axis = 0)
#         y_train_idx_sorted = y_train_idx_sorted[:30000]
#         y_train = y_train[y_train_idx_sorted].squeeze().reshape(-1,1)
#         X_train = X_train[y_train_idx_sorted].squeeze()
# =============================================================================
    print("Fitting GPR sklearn in run " + str(i))
    
    start = time.time()
    #set kernel
    kernel = TanimotoSK()
    #build and optimize GPR
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1.0)
    gpr.fit(X_train, y_train)
    end = time.time() - start
    print("Fitting time: ", end)


    print("Predicting GPR in run ", i)
    start = time.time()
    
    #as for the training set, also the samples used for inference cannot
    #exceed moderate numbers at a time. Therefore, the samples are batched
    #and predicted sequentially
    x_rest_batched = batch_data(x_rest, 512)

    predictions = []
    variances = []
    #actual_values = []
    for x in x_rest_batched:
        t = x.astype(np.float64)
        y_pred, y_var =gpr.predict(t, return_std=True)
        #y_pred = y_scaler.inverse_transform(y_pred) --> inverse transform not necessary?
        predictions.append(y_pred)
        variances.append(y_var)
    predictions = np.concatenate(predictions)
    variances = np.concatenate(variances)
    end = time.time() - start
    print("Prediction time: ", end)
    
    return predictions, variances


#-----------------------------------------------------------------------------# 

regression_functions_dict = {"lgbm": regression_lgbm, "gpr": regression_gpr_sklearn,
                             "gpr_gpflow": regression_gpr, "svr": regression_svr,
                             "dnn":regression_dnn}

