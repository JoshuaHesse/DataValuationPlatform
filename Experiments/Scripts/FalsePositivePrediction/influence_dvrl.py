#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:21:14 2023

@author: joshua

This file contains the function used to predict false positives via data valuation
using reinforcement learning. The file further imports the dvrl file from the 
github of the original publication.
"""
import pandas as pd
import numpy as np
from utils_jh import process_ranking, bedroc_score, enrichment_factor_score, get_scaffold_rate, run_logger, get_labels, get_labels_val
from sklearn.metrics import precision_score
import lightgbm
import dvrl
from typing import Tuple
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from rdkit import Chem

def run_dvrl( 
    db_train: pd.DataFrame,
    db_val: pd.DataFrame,
    name: str,
    representation: str = "ecfp",
    replicates: int = 5,
    log_predictions: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
    

    """Executes DVRL analysis on given dataset

    Uses the DVRL (Data Valuation using Reinforcement Learning) package to 
    train a LGBM Model and perform reinforcement learning to
    determine the importances of samples within the training set. The training samples are then ranked according to 
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

    tf.autograph.set_verbosity(0)
  
    #getting mols for train and val
    mols_train = list(db_train["SMILES"])
    mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
    mols_val = list(db_val["SMILES"])
    mols_val = [Chem.MolFromSmiles(x) for x in mols_val]
    
    #loading in the molecular representations according to the "representation
    #argument; molecular representations were calculated and stored via the 
    #"descr_export_pipeline_jh.py
    train_path = "../Datasets_descr/" + name + "/" + name + "_" + representation + "_train.pkl"
    x_train = pd.read_pickle(train_path)
    x_train = x_train.to_numpy()
    val_path = "../Datasets_descr/" + name + "/" + name + "_" + representation + "_val.pkl"
    x_val = pd.read_pickle(val_path)
    x_val = x_val.to_numpy()
    
    #the corresponding y_labels are calculated
    y_p_train, y_c_train, y_f_train, idx_train = get_labels(db_train)


    y_p_val, y_c_val, y_f_val, idx_val = get_labels_val(db_val)
    
    #assuring the correct shape of the imported arrays via the
    #expected vector size according to the representation, and the 
    #expected matrix length according to the corresponding y_labels 
    if(representation == "rdkit"):
        assert x_train.shape ==(len(y_p_train), 208), "x_train has wrong shape"
        assert x_val.shape ==(len(y_p_val), 208), "x_val has wrong shape"
    elif(representation == "ecfp"):
        assert x_train.shape ==(len(y_p_train), 1024), "x_train has wrong shape"
        assert x_val.shape ==(len(y_p_val), 1024), "x_val has wrong shape"

    
    #renaming for ease of reading
    y_train = y_p_train
    y_val = y_c_val

    #initialize result containers
    temp = np.zeros((replicates,9))
    logs = pd.DataFrame([])
    logs_temp = pd.DataFrame([])
    
    
    
    
    # Network parameters for the DVRL model
    parameters = dict()
    parameters['hidden_dim'] = 100
    parameters['comb_dim'] = 10
    parameters['iterations'] = 1000
    parameters['activation'] = tf.nn.relu
    parameters['layer_number'] = 5
    parameters['batch_size'] = 5000
    parameters['learning_rate'] = 0.01
    
    print("Beginning calculations")
    #iteratively train and measure importances according to number of replicates
    for i in range(replicates):
        tf.reset_default_graph()
        #set timer to determine time per iteration
        start = time.time()
        
        # Sets checkpoint file name
        checkpoint_file_name = './tmp/model.ckpt'

        

# =============================================================================
# 
#         num_pos = y_val.sum()
#         num_neg = len(y_val) - num_pos
#         weights_pos_sqrt = np.sqrt(1/(num_pos/(num_neg + num_pos)))
#         weights_neg_sqrt = np.sqrt(1/(num_neg/(num_neg + num_pos)))
#         class_weight = {0: weights_neg_sqrt, 1: weights_pos_sqrt}
# =============================================================================
        # Defines predictive model and problem type
        pred_model = lightgbm.LGBMClassifier(n_jobs=50, random_state = i)
        #pred_model = linear_model.LogisticRegression(max_iter=500, solver='lbfgs')
        problem = 'classification'

        # Flags for using stochastic gradient descent / pre-trained model
        flags = {'sgd': False, 'pretrain': False}

        # Initalizes DVRL: x_val and y_val are the confirmatory datapoints
        dvrl_class = dvrl.Dvrl(x_train, y_train, x_val, y_val, 
                               problem, pred_model, parameters, checkpoint_file_name, flags)

        # Trains DVRL using the metric specified in brackets
        dvrl_class.train_dvrl('auc')

        print('Finished dvrl training run number ' + str(i+1) + " of " + str(replicates))

        # Estimates data values
        importances = dvrl_class.data_valuator(x_train, y_train)
      

        #FP and TP labels are created from the importances 
        flags, flags_alt = process_ranking(y_train, importances)
        #the lapsed time is documented
        temp[i,0] = time.time() - start
        
        #get precision@90 for FPs and TPs
        temp[i,1] = precision_score(y_f_train, flags_alt[idx_train])
        temp[i,2] = precision_score(y_c_train, flags[idx_train])

        #get EF10 for FPs and TPs
        temp[i,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
        temp[i,4] = enrichment_factor_score(y_c_train, flags[idx_train])

        #get BEDROC20 for FPs and TPs
        temp[i,5] = bedroc_score(y_f_train, importances[idx_train], reverse=False)
        temp[i,6] = bedroc_score(y_c_train, importances[idx_train], reverse=True)

        #get diversity for FPs
        idx_fp = np.where(flags_alt == 1)[0]
        mols_fp = [mols_train[x] for x in idx_fp]
        temp[i,7] = get_scaffold_rate(mols_fp)

        #get diversity for TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols_train[x] for x in idx_tp]
        temp[i,8] = get_scaffold_rate(mols_tp)

        #optionally fill up logger
        if log_predictions is True:
            logs_temp = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                       flags[idx_train], importances[idx_train], "DVRL")
          #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
          #i replicates
            logs = pd.concat([logs, logs_temp], axis=0, ignore_index=True)

    return temp, logs
