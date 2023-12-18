#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:13:08 2023

@author: joshua

The function in this file is used to calculate predict false positives using 
catboost and its built-in object_importance function
"""
from catboost import Pool
from catboost import CatBoostClassifier as clf
import pandas as pd
import numpy as np
from rdkit import Chem
from utils_jh import get_labels, get_labels_val, process_ranking, bedroc_score, enrichment_factor_score, get_scaffold_rate, run_logger
from sklearn.metrics import precision_score
from typing import Tuple
import time

def run_catboost(
        db_train: pd.DataFrame,
        db_val: pd.DataFrame,
        name: str,
        replicates: int = 5,
        representation: str = "ecfp",
        log_predictions: bool = True
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes CatBoost analysis on given dataset

    Uses CatBoost object importance function to rank primary screen actives
    in terms of likelihood of being false positives or true positives. Unlike
    for MVS-A, top ranked compounds are TPs, bottom ranked compounds FPs. Finally,
    the function computes precision@90, EF10, BEDROC20 and scaffold diversity metrics.
    
    Args:
        mols:               (M,) mol objects from primary data
        x:                  (M, 1024) ECFPs of primary screen molecules
        y_p:                (M,) primary screen labels
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        replicates:         number of replicates to use for the run
        log_predictions:    enables raw predictions logging
     
    Returns:
        Tuple containing one array (1,9) with precision@90 for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """
    mols_train = list(db_train["SMILES"])
    mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
    mols_val = list(db_val["SMILES"])
    mols_val = [Chem.MolFromSmiles(x) for x in mols_val]
    
    #loading in the molecular representations according to the "representation
    #argument; molecular representations were calculated and stored via the 
    #"descr_export_pipeline_jh.py
    train_path = "../../../Datasets_descr/" + name + "/" + name + "_" + representation + "_train.pkl"
    x_train = pd.read_pickle(train_path)
    x_train = x_train.to_numpy()
    val_path = "../../../Datasets_descr/" + name + "/" + name + "_" + representation + "_val.pkl"
    x_val = pd.read_pickle(val_path)
    x_val = x_val.to_numpy()
    
    #the corresponding y_labels are extracted from the dataframes
    y_p_train, y_c_train, y_f_train, idx_train = get_labels(db_train)
  
   

    y_p_val, y_c_val, y_f_val, idx_val = get_labels_val(db_val)
    

    #create results containers
    #initialize result containers
    temp_self = np.zeros((replicates,9))
    temp_test = np.zeros((replicates,9))
    logs_self = pd.DataFrame([])
    logs_self_temp = pd.DataFrame([])
    logs_test= pd.DataFrame([])
    logs_test_temp = pd.DataFrame([])
    
    #loop analysis over replicates
    for j in range(replicates):
        print("[eval]: Catboost replicate ", j)
        #train catboost model, get sample importance and time it
        start = time.time()
        model = clf(iterations=100, random_seed=j)
        model.fit(x_train, y_p_train, verbose=False)
        idx_act = np.where(y_p_train==1)[0]
        pool1 = Pool(x_train, y_p_train)
        pool2 = Pool(x_train[idx_act], y_p_train[idx_act])
        pool3 = Pool(x_val, y_c_val)
        idx_cat_self, scores_self = model.get_object_importance(pool =pool2, 
                                                                train_pool =pool1)
        idx_cat_test, scores_test = model.get_object_importance(pool =pool3, 
                                                                train_pool =pool1)
        
        temp_self[j,0] = time.time() - start
        
        #rearrange output into a single vector, then binarize importance
        #scores (bottom 10% -> FP, top 90% TP)
        vals_self = np.empty((y_p_train.shape[0]))
        vals_self[idx_cat_self] = scores_self
        flags, flags_alt = process_ranking(y_p_train, vals_self)

        #get precision@90 for FP and TP retrieval
        temp_self[j,1] = precision_score(y_f_train, flags_alt[idx_train])
        temp_self[j,2] = precision_score(y_c_train, flags[idx_train])   
    
        #get EF10 for FPs and TPs
        temp_self[j,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
        temp_self[j,4] = enrichment_factor_score(y_c_train, flags[idx_train])

        #get BEDROC20 for FPs and TPs
        temp_self[j,5] = bedroc_score(y_f_train, vals_self[idx_train], reverse=False)
        temp_self[j,6] = bedroc_score(y_c_train, vals_self[idx_train], reverse=True)	
        
        #get scaffold diversity for compounds that got flagged as FPs
        idx_fp= np.where(flags_alt == 1)[0]
        mols_fp = [mols_train[x] for x in idx_fp]
        temp_self[j,7] = get_scaffold_rate(mols_fp)
        
        #get scaffold diversity for compounds that got flagged as TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols_train[x] for x in idx_tp]
        temp_self[j,8] = get_scaffold_rate(mols_tp)

        #optionally fill up logger
        if log_predictions is True:
             logs_self_temp = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                        flags[idx_train], vals_self[idx_train], "Catboost_self")
           #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
           #i replicates
             logs_self = pd.concat([logs_self, logs_self_temp], axis=0, ignore_index=True)
             
        
        temp_test[j,0] = time.time() - start
        
        #rearrange output into a single vector, then binarize importance
        #scores (bottom 10% -> FP, top 90% TP)
        vals_test = np.empty((y_p_train.shape[0]))
        vals_test[idx_cat_test] = scores_test
        flags, flags_alt = process_ranking(y_p_train, vals_test)

        #get precision@90 for FP and TP retrieval
        temp_test[j,1] = precision_score(y_f_train, flags_alt[idx_train])
        temp_test[j,2] = precision_score(y_c_train, flags[idx_train])   
    
        #get EF10 for FPs and TPs
        temp_test[j,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
        temp_test[j,4] = enrichment_factor_score(y_c_train, flags[idx_train])

        #get BEDROC20 for FPs and TPs
        temp_test[j,5] = bedroc_score(y_f_train, vals_test[idx_train], reverse=False)
        temp_test[j,6] = bedroc_score(y_c_train, vals_test[idx_train], reverse=True)	
        
        #get scaffold diversity for compounds that got flagged as FPs
        idx_fp = np.where(flags_alt == 1)[0]
        mols_fp = [mols_train[x] for x in idx_fp]
        temp_test[j,7] = get_scaffold_rate(mols_fp)
        
        #get scaffold diversity for compounds that got flagged as TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols_train[x] for x in idx_tp]
        temp_test[j,8] = get_scaffold_rate(mols_tp)

        #optionally fill up logger
        if log_predictions is True:
             logs_test_temp = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                        flags[idx_train], vals_test[idx_train], "Catboost_test")
           #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
           #i replicates
             logs_test = pd.concat([logs_test, logs_test_temp], axis=0, ignore_index=True)
    
    return temp_self, temp_test, logs_self, logs_test