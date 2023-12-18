#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:21:24 2023

@author: joshua
The function in this file is used to predict false positives using data shapley, 
approximating shapley values using KNN.
"""

from typing import Tuple
from utils_jh import get_labels, get_labels_val, process_ranking, bedroc_score, enrichment_factor_score, get_scaffold_rate, run_logger
import numpy as np
from sklearn.metrics import precision_score
import time
from rdkit import Chem
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from datascope.importance.common import SklearnModelAccuracy
from datascope.importance.shapley import ShapleyImportance
import imblearn.under_sampling as im
import lightgbm

def run_knn(db_train: pd.DataFrame,
        db_val: pd.DataFrame,
        name: str,
        representation: str,
        replicates: int,
        log_predictions: bool = True
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    
        """Executes KNN analysis on given dataset
    
        uses KNN with k = 5 and the KNN direct shapley importance calculations from 
        datascope to determine importances of a subset of samples from the training set by using
        random undersampling at a ratio of 1/5 for actives/inactives. This is done in replicates.
        The validation set is used to calculate importances via the y_c_val labels, 
        the confirmatory labels of the validation set. The samples are then ranked according to 
        importances. Top 10% are defined as proponents, bottom 10% as opponents. Finally, the 
        function computes precision@90, EF10, BEDROC20 and scaffold diversity metrics. 
        If log_predictions are set to TRUE, each replicate is logged and they are concatenated
        to a final logs variable.
        
        Args:
            db_train:           (M, 5) DataFrame of the training set for the resepective dataset as read directly from the .csv file 
            db_val:             (N, 5) DataFrame of the training set for the resepective dataset as read directly from the .csv file 
            name:               name of the dataset (to load in the precalculated molecular representations)
            representation:     type of molecular representation to be used [ecfp, rdkit, smiles]
            replicates          number of replicates, meaning number of undersampling runs, using different random_state variables
            log_predictions:    enables raw predictions logging
         
        Returns:
            Tuple containing one array (1,9) with precision@90 for FP and TP retrieval, 
            scaffold diversity and training time, and one dataframe (V,5) with
            SMILES, true labels and raw predictions (logs)
        
            """
        #get the mols for the validation set as well as the corresponding labels
        mols_val = list(db_val["SMILES"])
        mols_val = [Chem.MolFromSmiles(x) for x in mols_val]
        y_p_val, y_c_val, y_f_val, idx_val = get_labels_val(db_val)
        
        #define the type of classifier
        neigh = KNeighborsClassifier(n_neighbors=5)
        
        #get the initial labels for the full training set to use y_p_train for undersampling
        y_p_train, y_c_train, y_f_train, idx_train = get_labels(db_train)
        
        
        #initialize result containers
        temp = np.zeros((replicates,9))
        logs = pd.DataFrame([])
        logs_temp = pd.DataFrame([])
        
        #load in precalculated representations (either ecfp or rdkit)
        train_path = "../../../Datasets_descr/" + name + "/" + name + "_" + representation + "_train.pkl"
        x_train = pd.read_pickle(train_path)
        val_path = "../../../Datasets_descr/" + name + "/" + name + "_" + representation + "_val.pkl"
        x_val = pd.read_pickle(val_path)
        
        
        #assuring the correct shape of the imported arrays
        if(representation == "rdkit"):
            assert x_train.shape ==(len(y_p_train), 208), "x_train has wrong shape"
            assert x_val.shape ==(len(y_p_val), 208), "x_train has wrong shape"
        elif(representation == "ecfp"):
            assert x_train.shape ==(len(y_p_train), 1024), "x_train has wrong shape"
            assert x_val.shape ==(len(y_p_val), 1024), "x_train has wrong shape"

        #iteratively train and measure importances on small subset of training set using random undersampling
        for i in range(replicates):
           
            #timing the method for subsequent analysis
             
            start = time.time()


         
            #random undersampler generates subset of the the training set with a ratio of primary
            #actives to primary inactives of 1/5 by using the y_p_train labels
            rus = im.RandomUnderSampler(sampling_strategy=0.2, random_state=i)
            
            #the undersampling is done on the db_train (M, 5) to generate a new 
            #variable train_under with (B, 5) where B<<M
            #--> this is done on the db_train to later access the mol_substructures
            train_under_dp, y_p_train_under_dp = rus.fit_resample(db_train, y_p_train)
            
            
            #the same undersampler is used to subsample the preloaded representations 
            x_train_under, y_p_train_under = rus.fit_resample(x_train, y_p_train)
            #train_under is then used as if it were the entire dataset for the rest of this iteration
            y_p_train_under, y_c_train_under, y_f_train_under, idx_train_under = get_labels(train_under_dp)
            
            
            mols_train_under = list(train_under_dp["SMILES"])
            mols_train_under = [Chem.MolFromSmiles(x) for x in mols_train_under]
            
            
           
            
            
            
            #use KNN datascope importance measurement to measure sample importance by using the 
            #validation set for scoring, here applying y_c_val, not y_p_val!
            pipe_ecfp = Pipeline([('model', neigh)])
            utility = SklearnModelAccuracy(pipe_ecfp)
            importance = ShapleyImportance(method="neighbor", utility=utility)
            importances = importance.fit(x_train_under, y_p_train_under).score(x_val, y_c_val)

            #the lapsed time is documented
            temp[i,0] = time.time() - start
            
            #FP and TP labels are created from the importances 
            flags, flags_alt = process_ranking(y_p_train_under, importances)

            #get precision@90 for FPs and TPs
            temp[i,1] = precision_score(y_f_train_under, flags_alt[idx_train_under])
            temp[i,2] = precision_score(y_c_train_under, flags[idx_train_under])
            
            #get EF10 for FPs and TPs
            temp[i,3] = enrichment_factor_score(y_f_train_under, flags_alt[idx_train_under])
            temp[i,4] = enrichment_factor_score(y_c_train_under, flags[idx_train_under])
            
            #get BEDROC20 for FPs and TPs
            temp[i,5] = bedroc_score(y_f_train_under, importances[idx_train_under], reverse=False)
            temp[i,6] = bedroc_score(y_c_train_under, importances[idx_train_under], reverse=True)
            
            #get diversity for FPs
            idx_fp = np.where(flags_alt == 1)[0]
            mols_fp = [mols_train_under[x] for x in idx_fp]
            temp[i,7] = get_scaffold_rate(mols_fp)
                
            #get diversity for TPs
            idx_tp = np.where(flags == 1)[0]
            mols_tp = [mols_train_under[x] for x in idx_tp]
            temp[i,8] = get_scaffold_rate(mols_tp)
             
            #optionally fill up logger
            if log_predictions is True:
               logs_temp = run_logger(mols_train_under, idx_train_under, y_f_train_under, y_c_train_under, flags_alt[idx_train_under],
                                           flags[idx_train_under], importances[idx_train_under], "knn")
              #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
              #i replicates
               logs = pd.concat([logs, logs_temp], axis=0, ignore_index=True)
            
        return temp, logs
