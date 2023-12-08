#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:20:21 2023

@author: joshua

This file contains supplementary functions for the regression functions used
for active learning´
"""
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

def build_and_compile_model(
        input_shape: int,
        seed:int = 0
        )-> tf.keras.Model:
    """
    

    Parameters
    ----------
    input_shape : int
        Dimensions of the molecular fingerprints, e.g. 1024 for ecfp

    Returns
    -------
    model : TYPE
        tensroflow model for regression, e.g. influence prediction
        
    ------
    
    builds and compiles a simple 2-hidden-layer DNN for influence regression

    """
    
    tf.random.set_seed(seed)
    model = keras.Sequential([
        layers.Dense(64, input_shape = (input_shape,), activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

#-----------------------------------------------------------------------------#
def batch_data(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]