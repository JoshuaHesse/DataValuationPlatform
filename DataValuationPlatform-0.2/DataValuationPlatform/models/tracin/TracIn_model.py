#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:36:33 2023

@author: joshuahesse
"""
from DataValuationPlatform.models.model_class import Model
import DataValuationPlatform.models.tracin.tracin_utils as tu
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import time
import sys
import os
from shutil import rmtree

class TracIn(Model):
    def __init__(self, n_jobs=30):
        super().__init__(n_jobs) 
        self.algorithm_name = "TracIn"

    def calculate_influence_smiles(self, dataset, influence_type, seed = 0):
        x_train, x_val, training_set, validation_set, chars_to_int = tu.vectorize_smiles(dataset.training_set, dataset.validation_set, max_length=100)
        y_train_primary = dataset.training_set_labels
        y_val_conflirmatory = dataset.validation_set_labels
        
        rus = RandomUnderSampler(sampling_strategy=0.1, random_state=seed)
        idx_under, y_train_under = rus.fit_resample(np.reshape(range(len(y_train_primary)), (-1, 1)), y_train_primary)
        x_train_under = x_train[idx_under]
        x_train_under = np.squeeze(x_train_under)
        y_train_under = np.squeeze(y_train_under)
        x_train_under, y_train_under = shuffle(x_train_under, y_train_under)
        
        input_shape = (x_train_under.shape[1], x_train_under.shape[2]) 
        training_set = tf.data.Dataset.from_tensor_slices((x_train, y_train_primary))
        batched_training_set = training_set.batch(batch_size=512)
        
        start_time1 = time.time()
        checkpoint_path = "TracIn" + dataset.dataset_name +"_/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        progress_callback = tu.ProgressCallback()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=0,
            save_weights_only=True,
            save_freq= "epoch")
        
        self.model = tu.create_lstm_model(input_shape, random_state = seed)
        print("fitting basic model")
        self.model.save_weights(checkpoint_path.format(epoch=0))
        start = time.time()
        self.model.fit(x_train,
                        y_train_primary,
                        epochs=100,
                        batch_size=512,
                        callbacks=[cp_callback,progress_callback],
                        validation_split = 0.1,
                        verbose=0)
        end = time.time() - start
        print("Fitting time in seconds: ", end)
        self.models = []
        for i in [30, 60, 90]:
            basic_model = tu.create_lstm_model(input_shape,random_state = seed)
            basic_model.load_weights(checkpoint_path.format(epoch=i)).expect_partial()
            self.models.append(basic_model)
        
        if influence_type.lower() == "self":
            results = tu.get_self_influence(batched_training_set, self.models)
            self.influence_scores = results.get("self_influences") * (-1)
            end = time.time() - start_time1
            print("TracIn_self influence score calculation for " + str(len(y_train_primary)) + " training samples of " + dataset.dataset_name + " done; time in seconds: " +str(round(end, 2)))
        elif influence_type.lower() == "test":
            results = tu.get_test_influence(batched_training_set, x_val, y_val_conflirmatory, self.models)
            self.influence_scores = results.get("test_influences") * (-1)
            end = time.time() - start_time1
            print("TracIn_test influence score calculation for " + str(len(y_train_primary)) + " training samples of " + dataset.dataset_name + " done; time in seconds: " +str(round(end, 2)))
        else:
            print("Unknown influence type. Please chose from [self, test]")
        rmtree(checkpoint_dir)
        return self.influence_scores 
    
    def calculate_influence(self, dataset, influence_type="self",seed = 0):
         
        if dataset.representation.lower() == "smiles":
            return self.calculate_influence_smiles(dataset, influence_type)

        y_train_primary = dataset.training_set_labels
        x_train = dataset.training_set_descriptors
        
        y_val_conflirmatory = dataset.validation_set_labels
        x_val = dataset.validation_set_descriptors

        training_set = tf.data.Dataset.from_tensor_slices((x_train, y_train_primary))
        batched_training_set = training_set.batch(batch_size=512)
        
        start_time1 = time.time()
        checkpoint_path = "TracIn" + dataset.dataset_name +"_/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        progress_callback = tu.ProgressCallback()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=0,
            save_weights_only=True,
            save_freq= "epoch")
        
        self.model = tu.make_model(x_train = x_train,random_state = seed)
        print("fitting basic model")
        self.model.save_weights(checkpoint_path.format(epoch=0))
        start = time.time()
        self.model.fit(x_train,
                        y_train_primary,
                        epochs=100,
                        batch_size=512,
                        callbacks=[cp_callback,progress_callback],
                        validation_split = 0.1,
                        verbose=0)
        end = time.time() - start
        print("Fitting time in seconds: ", end)
        self.models = []
        for i in [30, 60, 90]:
            basic_model = tu.make_model(x_train = x_train,random_state = seed)
            basic_model.load_weights(checkpoint_path.format(epoch=i)).expect_partial()
            self.models.append(basic_model)
        
        if influence_type.lower() == "self":
            results = tu.get_self_influence(batched_training_set, self.models)
            self.influence_scores = results.get("self_influences") * (-1)
            end = time.time() - start_time1
            print("TracIn_self influence score calculation for " + str(len(y_train_primary)) + " training samples of " + dataset.dataset_name + " done; time in seconds: " +str(round(end, 2)))
        elif influence_type.lower() == "test":
            results = tu.get_test_influence(batched_training_set, x_val, y_val_conflirmatory, self.models)
            self.influence_scores = results.get("test_influences") * (-1)
            end = time.time() - start_time1
            print("TracIn_test influence score calculation for " + str(len(y_train_primary)) + " training samples of " + dataset.dataset_name + " done; time in seconds: " +str(round(end, 2)))
        else:
            print("Unknown influence type. Please chose from [self, test]")
        rmtree(checkpoint_dir)
        return self.influence_scores 
    
        
