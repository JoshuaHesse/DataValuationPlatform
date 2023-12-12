#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:52:15 2023

@author: joshua

This file contains all functions used for active learning via TracIn, either 
calculating the influence of training samples on their own prediction, a
validation sets prediction, or a validation sets prediction only incorporating 
active samples.
"""

import numpy as np
import warnings
import time

#tensorflow isn't present in the datascope environment,
#and would throw errors otherwhise
try:
    import tensorflow as tf
    from tracin_utils import make_model, get_test_influence, get_self_influence
except ImportError:
    warnings.warn('TracIn tensorflow failed to import', ImportWarning)
import os
from shutil import rmtree



#-----------------------------------------------------------------------------#  

def influences_tracin(
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
  x_val: np.ndarray (N,D)
      Vectorized molecular representations of N validation molecules, D dimensions 
      per molecule
 y_val : np.ndarray (N, 1)
     Labels for N validation molecules, binary
    i : int
        ith step of iteration.
    

    Returns
    -------
    influences: np.ndarray (M, 1)
        Influences for all M molecules calculated via TracIn
    
    -----
    
    calculates influence scores of training samples via TracIn, a gradient 
    tracing function applied to deep neural networks. The function trains a
    DNN and exports the learned weights at the end of every epoch. Afterwards,
    a few checkpoints of different timepoints during training are reloaded and
    are used to calculate the influence of every sample on its own prediction
    at these checkpoints.
    
    The influences of the training samples are calculated via the effects on 
    the gradients of a separate validation set 
    """
    
    
    #x_train and y_train have to be split into a batched dataset,
    #as tracin calculates gradients for all samples causing big matrices
    #for too many samples
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    batched_dataset = dataset.batch(batch_size=512)
    
    #weights are temporarily saved in checkpoint files
    checkpoint_path = "active_learning_batch/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq= "epoch")

    # Create a new model instance
    basic_model = make_model(x_train = x_train)
    print("fitting basic model in run ", i)
    # Save the weights using the `checkpoint_path` format

    basic_model.save_weights(checkpoint_path.format(epoch=0))
    start = time.time()
    # Train the model with the new callback
    basic_model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=512,
                    callbacks=[cp_callback],
                    validation_split = 0.1,
                    verbose=0)
    end = time.time() - start
    print("Fitting time: ", end)
    start = time.time()
    basic_models = []
    
    #models recovered from the checkpoints to calculate gradients at different 
    #points in the fitting process
    for t in [30,60,90]:
        basic_model = make_model(x_train = x_train)
        basic_model.load_weights(checkpoint_path.format(epoch=t)).expect_partial()
        basic_models.append(basic_model)
    end = time.time() - start



    print("Running TracIn in run ", i)
    start = time.time()
    #influences of all samples on the validation set prediction are calculated
    trackin_train_test_influences = get_test_influence(batched_dataset,x_val, y_val, basic_models)
    end = time.time() - start
    print("TracIn calculation time: " + str(end)+" in step " + str(i))
    tf.keras.backend.clear_session()
    #checkpoint_files are deleted
    rmtree(checkpoint_dir)
    
    #*-1 had to be done during FP detection for consistency with other 
    #methods and is kept here for consistency over applications
    return trackin_train_test_influences.get("test_influences") * -1

#-----------------------------------------------------------------------------#


def influences_tracin_self(
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
 i : int
      ith step of iteration. The default is 0.´
 seed: int, optional
    seed to set random seed of mvsa. The default is 0. 

    Returns
    -------
    influences: np.ndarray (M, 1)
        Influences for all M molecules calculated via TracIn
    
    -----
    
    calculates influence scores of training samples via TracIn, a gradient 
    tracing function applied to deep neural networks. The function trains a
    DNN and exports the learned weights at the end of every epoch. Afterwards,
    a few checkpoints of different timepoints during training are reloaded and
    are used to calculate the influence of every sample on its own prediction
    at these checkpoints.
    """
    
    
    #x_train and y_train have to be split into a batched dataset,
    #as tracin calculates gradients for all samples causing big matrices
    #for too many samples
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    batched_dataset = dataset.batch(batch_size=512)
    
    #weights are temporarily saved in checkpoint files
    checkpoint_path = "active_learning_batch/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq= "epoch")

    # Create a new model instance
    basic_model = make_model(x_train = x_train, random_state=seed)
    print("fitting basic model in run ", i)
    # Save the weights using the `checkpoint_path` format

    basic_model.save_weights(checkpoint_path.format(epoch=0))
    start = time.time()
    # Train the model with the new callback
    basic_model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=512,
                    callbacks=[cp_callback],
                    validation_split = 0.1,
                    verbose=0)
    end = time.time() - start
    print("Fitting time: ", end)
    start = time.time()
    basic_models = []
    
    #models recovered from the checkpoints to calculate gradients at different 
    #points in the fitting process
    for t in [30,60,90]:
        basic_model = make_model(x_train = x_train, random_state=seed)
        basic_model.load_weights(checkpoint_path.format(epoch=t)).expect_partial()
        basic_models.append(basic_model)
    end = time.time() - start



    print("Running TracIn in run ", i)
    start = time.time()
    #influences of all samples on their own predictions are calculated
    trackin_train_self_influences = get_self_influence(batched_dataset, basic_models)
    end = time.time() - start
    print("TracIn_self calculation time: " + str(end)+" in step " + str(i))
    tf.keras.backend.clear_session()
    #checkpoint_files are deleted
    rmtree(checkpoint_dir)
    
    #*-1 had to be done during FP detection for consistency with other 
    #methods and is kept here for consistency over applications
    return trackin_train_self_influences.get("self_influences") * -1

#-----------------------------------------------------------------------------#

def influences_tracin_pos(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        i: int,
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
  x_val: np.ndarray (N,D)
      Vectorized molecular representations of N validation molecules, D dimensions 
      per molecule
 y_val : np.ndarray (N, 1)
     Labels for N validation molecules, binary
 i : int
     ith step of iteration.
 seed: int, optional
    seed to set random seed of mvsa. The default is 0.    

    Returns
    -------
    influences: np.ndarray (M, 1)
        Influences for all M molecules calculated via TracIn
    
    -----
    
    calculates influence scores of training samples via TracIn, a gradient 
    tracing function applied to deep neural networks. The function trains a
    DNN and exports the learned weights at the end of every epoch. Afterwards,
    a few checkpoints of different timepoints during training are reloaded and
    are used to calculate the influence of every sample on its own prediction
    at these checkpoints.
    
    The influences of the training samples are calculated via the effects on 
    the gradients of a separate validation set. But, unlike influences_tracin,
    only the active sampels of the validation set are used.
    """
    
    #identifying the active samples in the validation set
    idx_pos = []
    for p in range(len(y_val)):
        if y_val[p]==1:
            idx_pos.append(p)
        
    x_val_pos = x_val[idx_pos]
    y_val_pos = y_val[idx_pos]
    #x_train and y_train have to be split into a batched dataset,
    #as tracin calculates gradients for all samples causing big matrices
    #for too many samples
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    batched_dataset = dataset.batch(batch_size=512)
    
    #weights are temporarily saved in checkpoint files
    checkpoint_path = "active_learning_batch/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq= "epoch")

    # Create a new model instance
    basic_model = make_model(x_train = x_train, random_state=seed)
    print("fitting basic model in run ", i)
    # Save the weights using the `checkpoint_path` format

    basic_model.save_weights(checkpoint_path.format(epoch=0))
    start = time.time()
    # Train the model with the new callback
    basic_model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=512,
                    callbacks=[cp_callback],
                    validation_split = 0.1,
                    verbose=0)
    end = time.time() - start
    print("Fitting time: ", end)
    start = time.time()
    basic_models = []
    
    #models recovered from the checkpoints to calculate gradients at different 
    #points in the fitting process
    for t in [30,60,90]:
        basic_model = make_model(x_train = x_train, random_state=seed)
        basic_model.load_weights(checkpoint_path.format(epoch=t)).expect_partial()
        basic_models.append(basic_model)
    end = time.time() - start



    print("Running TracIn in run ", i)
    start = time.time()
    #influences of all samples on the prediction of positive samples of the 
    #validation set are calculated
    trackin_train_test_influences = get_test_influence(batched_dataset,x_val_pos,y_val_pos, basic_models)
    end = time.time() - start
    print("TracIn_pos calculation time: " + str(end)+" in step " + str(i))
    tf.keras.backend.clear_session()
    #checkpoint_files are deleted
    rmtree(checkpoint_dir)
    
    #*-1 had to be done during FP detection for consistency with other 
    #methods and is kept here for consistency over applications
    return trackin_train_test_influences.get("test_influences") * -1

