#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:37:02 2023

@author: joshuahesse
"""

from catboost import Pool
from catboost import CatBoostClassifier as clf   
from DataValuationPlatform.models.model_class import Model 
import time
import numpy as np
        
class CatBoost(Model):
    def __init__(self, n_jobs=30):
        super().__init__(n_jobs) 
        self.algorithm_name = "CatBoost"
    def calculate_influence(self, dataset, influence_type="self",seed = 0):
        
        y_train_primary = dataset.training_set_labels
        x_train = dataset.training_set_descriptors
        
        y_val_confirmatory = dataset.validation_set_labels
        x_val = dataset.validation_set_descriptors

        start = time.time()
        self.model = clf(iterations=100, random_seed=seed, thread_count = self.n_jobs)
        self.model.fit(x_train, y_train_primary, verbose=False)
        idx_act = np.where(y_train_primary==1)[0]
        idx_scores = None
        if influence_type.lower() == "self":
            pool1 = Pool(x_train, y_train_primary)
            pool2 = Pool(x_train[idx_act], y_train_primary[idx_act])
            idx_scores, scores_sorted = self.model.get_object_importance(pool =pool2, train_pool =pool1)
        elif influence_type.lower() == "test":
            pool1 = Pool(x_train, y_train_primary)
            pool2 = Pool(x_val, y_val_confirmatory)
            idx_scores, scores_sorted = self.model.get_object_importance(pool =pool2, train_pool =pool1)
        else:
            print("Unknown influence type. Please chose from [self, test]")
            return None
        self.influence_scores = np.empty((y_train_primary.shape[0]))
        self.influence_scores [idx_scores] = scores_sorted
        end = time.time() - start
        print(f"CatBoost {influence_type} influence score calculation for {str(len(y_train_primary))} training samples of {dataset.dataset_name} done; time in seconds: {str(round(end, 2))}")
        return self.influence_scores
        
