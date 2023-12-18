#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:37:26 2023

@author: joshuahesse
"""

from DataValuationPlatform.models.model_class import Model
import tensorflow as tf
import lightgbm
from DataValuationPlatform.models.dvrl import dvrl
class DVRL(Model):
    
    
    def __init__(self, n_jobs=30, parameters = "default", metric ="auc"):
        super().__init__(n_jobs)  
        if parameters == "default":
            self.parameters = {
                'hidden_dim': 100,
                'comb_dim': 10,
                'iterations': 1000,
                'activation': tf.nn.relu,
                'layer_number': 5,
                'batch_size': 5000,
                'learning_rate': 0.01,
                'inner_iterations': 100
            }
        elif isinstance(parameters, dict):
            self.parameters = parameters
        else:
            print("Please select the default parameters or feed a dictionary containing custom parameters")
        self.metric=metric
        self.algorithm_name = "DVRL"
            
    def calculate_influence(self, dataset,seed = 0):
        y_train_primary = dataset.training_set_labels
        x_train = dataset.training_set_descriptors
        
        y_val_confirmatory = dataset.validation_set_labels
        x_val = dataset.validation_set_descriptors
        
        checkpoint_file_name = './tmp/model.ckpt'
        self.model = lightgbm.LGBMClassifier(n_jobs=self.n_jobs, random_state = seed)
        problem = 'classification'
        flags = {'sgd': False, 'pretrain': False}
        dvrl_class = dvrl.Dvrl(x_train, y_train_primary, x_val, y_val_confirmatory, problem, self.model, self.parameters, checkpoint_file_name, flags)
        dvrl_class.train_dvrl(self.metric)
        self.influence_scores = dvrl_class.data_valuator(x_train, y_train_primary)
        return self.influence_scores
