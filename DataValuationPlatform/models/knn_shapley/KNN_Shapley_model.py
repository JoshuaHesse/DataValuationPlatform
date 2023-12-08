#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:28:27 2023

@author: joshuahesse
"""
from DataValuationPlatform.models.model_class import Model
from datascope.importance.common import SklearnModelAccuracy
from datascope.importance.shapley import ShapleyImportance
import imblearn.under_sampling as im
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

class KNN_Shapley(Model):
    def __init__(self, n_jobs=30):
        super().__init__(n_jobs) 
        self.algorithm_name = "KNN_Shapley"
  
    def calculate_influence(self, dataset,seed = 0):
        self.model = KNeighborsClassifier(n_neighbors=5)
        y_train_primary = np.array(dataset.training_set_labels)
        x_train = dataset.training_set_descriptors
        y_val_confirmatory = np.array(dataset.validation_set_labels)
        x_val = dataset.validation_set_descriptors

        rus = im.RandomUnderSampler(sampling_strategy=0.2, random_state=seed)
        idx, y_train_under = rus.fit_resample(np.array(range(len(y_train_primary))).reshape(-1,1), y_train_primary)
        x_train_under = x_train[idx].squeeze()
        pipe_ecfp = Pipeline([('model', self.model)])
        utility = SklearnModelAccuracy(pipe_ecfp)
        importance = ShapleyImportance(method="neighbor", utility=utility)
        self.influence_scores = np.zeros(len(y_train_primary))
        scores = importance.fit(x_train_under, y_train_under).score(x_val, y_val_confirmatory)
        self.influence_scores[idx] = np.array(scores).reshape(-1, 1)
        return self.influence_scores
