#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:36:08 2023

@author: joshuahesse
"""
from DataValuationPlatform.models.model_class import Model
import lightgbm as lgb
import numpy as np
from sklearn.metrics import *
from DataValuationPlatform.models.mvsa.MVS_A import sample_analysis 
import time

class MVSA(Model):
    def __init__(self, n_jobs=30):
        super().__init__(n_jobs) 
        self.algorithm_name = "MVSA"
  
    def calculate_influence(self, dataset,seed = 0):
        y_train_primary = dataset.training_set_labels
        x_train = dataset.training_set_descriptors

        obj = sample_analysis(x = x_train,
                              y = y_train_primary,
                              params = "default",
                              verbose = False,
                              seed = seed,
                              num_threads = self.n_jobs)
        obj.get_model()
        self.model = obj.model
        start = time.time()
        self.influence_scores = obj.get_importance()

        end = time.time() - start
        print("MVSA influence score calculation for " + str(len(y_train_primary)) + " training samples of " + dataset.dataset_name + " done; time in seconds: " +str(round(end, 2)))
        return self.influence_scores * (-1)

    
