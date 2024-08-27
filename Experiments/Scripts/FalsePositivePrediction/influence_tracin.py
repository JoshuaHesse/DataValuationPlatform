#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:48:44 2023

@author: joshua
"""
import gc
import pandas as pd
import numpy as np
import time
import datetime
import os
from imblearn.under_sampling import RandomUnderSampler
from shutil import rmtree
from rdkit import Chem
from typing import Tuple
from utils_jh import process_ranking, get_labels, get_labels_val, bedroc_score, enrichment_factor_score, get_scaffold_rate, run_logger
import tracin_utils
from sklearn.metrics import precision_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf


def run_tracin_importance_LSTM(
        train: pd.DataFrame,
        val: pd.DataFrame,
        name: str,
        replicates: int = 5,
        log_predictions: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    
    """Executes TracIn analysis on given dataset
    
    
   3 influence scores are computed: 
     - The first uses the self_influence definition,
       calculatiung the influence of a training sample on its own prediction by 
       squaring its gradients.
     - The second uses the test_influence definition,
       calculating the influence of a training sample on the prediction of the 
       test samples by multiplying its gradients with the gradients of all test
       samples, summing over the weights and averaging over the test samples.
     - The third method works the same as the second, but only using true positives
       as as test samples 
   
    
   The training samples are then ranked according to their influences, or "importances". Interestingly, the labels first have to be flipped
   (by multiplying with -1) to be of the same orientation as the other importance score methods. Top 10% are 
   defined as proponents, bottom 10% as opponents. Finally, the 
   function computes precision@90, EF10, BEDROC20 and scaffold diversity metrics. 
   If log_predictions are set to TRUE, each replicate is logged and they are concatenated
   to a final logs variable.
   
   This method uses the Smiles representations of the molecules for prediction by
   tokenizing and vectorizing them. The vectorized one hot encoded samples are then
   used to train an LSTM model.
    
    Args:
        db_train:           (M, 5) DataFrame of the training set for the resepective dataset as read directly from the .csv file 
        db_val:             (N, 5) DataFrame of the training set for the resepective dataset as read directly from the .csv file 
        name:               name of the dataset (to load in the precalculated molecular representations)
        replicates          number of replicates, meaning number of undersampling runs, using different random_state variables
        log_predictions:    enables raw predictions logging
     
    Returns:
        Tuple containing three arrays (1,9) with precision@90 for FP and TP retrieval, 
        scaffold diversity and training time, and three dataframes (V,5) with
        SMILES, true labels and raw predictions (logs); one of each for the normal TracIn implementation, one for the
        implementation only using TPs as the validation  set
    
    
    """
    #get mols for structure variance analysis
   
    
    #the smiles representations are extracted and vectorized
    #the d_train and val dataframes are given to the vectorize_smiles function
    #and returned, potentially altered. This is due to the fact that molecules with smiles
    #longer than max_length have to be dropped, not only in the training set matrix but in all 
    #samples; all labels, everywhre
    
    smiles_matrix_train, smiles_matrix_val, db_train, db_val, chars_to_int = tracin_utils.vectorize_smiles(train, val, max_length=100)
    
    #the corresponding y_labels are calculated
    mols_train = list(db_train["SMILES"])
    mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
    mols_val = list(db_val["SMILES"])
    mols_val = [Chem.MolFromSmiles(x) for x in mols_val]
    y_p_train, y_c_train, y_f_train, idx_train = get_labels(db_train)


    y_p_val, y_c_val, y_f_val, idx_val = get_labels_val(db_val)
    

    #The indices of all true positives from the validation set are extracted and
    #used to generate a TP-only validation set
    idx_pos = []
    for i in range(len(y_c_val)):
        if y_c_val[i]==1:
            idx_pos.append(i)
        
    x_val_pos = smiles_matrix_val[idx_pos]
    y_c_val_pos = y_c_val[idx_pos]
        


    #undersampling is used to improve dataset imbalance
    rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    idx_under, y_train_under = rus.fit_resample(np.reshape(range(len(y_p_train)), (-1, 1)), y_p_train)
    X_train_under = smiles_matrix_train[idx_under]
    X_train_under = np.squeeze(X_train_under)
    y_train_under = np.squeeze(y_train_under)
    X_train_under, y_train_under = shuffle(X_train_under, y_train_under)
    
    input_shape = (X_train_under.shape[1], X_train_under.shape[2]) 
    dataset = tf.data.Dataset.from_tensor_slices((smiles_matrix_train, y_p_train))
    batched_dataset = dataset.batch(batch_size=512)
    
    X_train, X_test, y_train, y_test = train_test_split( X_train_under, y_train_under, test_size=0.1, random_state=42)
    #for frequency of saving checkpoints:

    #initialize result containers

    temp = np.zeros((replicates,9))
    temp_pos = np.zeros((replicates,9))
    temp_self = np.zeros((replicates,9))
    logs = pd.DataFrame([])
    logs_temp = pd.DataFrame([])
    logs_pos = pd.DataFrame([])
    logs_temp_pos = pd.DataFrame([])
    logs_self = pd.DataFrame([])
    logs_temp_self = pd.DataFrame([])
    
    results = []
    results_pos_only = []
    results_self = []



    for r in range(replicates):

        start = time.time()
        checkpoint_path = "test_influence" + str(r) +"_" + name +"_/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        #cp1 = os.listdir(checkpoint_dir)[3][5:7]
        #cp2 = os.listdir(checkpoint_dir)[5][5:7]
        #cp2 = os.listdir(checkpoint_dir)[7][5:7]
        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq= "epoch")
    
        # Create a new model instance
        basic_model = tracin_utils.create_lstm_model(input_shape, random_state = r)
        print("fitting basic model in run ", r)
        # Save the weights using the `checkpoint_path` format
        #had to take the save at epoch 0 step out for model 7
        basic_model.save_weights(checkpoint_path.format(epoch=0))
        start_all = time.time()
        # Train the model with the new callback
        basic_model.fit(X_train, 
                  y_train,
                  epochs=100, 
                  batch_size=512, 
                  callbacks=[cp_callback],
                  validation_data= (X_test, y_test),
                  verbose=0)
        end = time.time() - start
        print("Fitting time: ", end)
        start = time.time()
        
        
        basic_models = []
        # when using RUS for some reason the last cp is at 89
        #when not using rus the last cp is at 90
        for i in [30, 60, 90]:
            basic_model = tracin_utils.create_lstm_model(input_shape, random_state = r)
            basic_model.load_weights(checkpoint_path.format(epoch=i)).expect_partial()
            basic_models.append(basic_model)
        end = time.time()
        print(datetime.timedelta(seconds=end - start))  

    
    
        print("TracIn Test-Influence on scaled model in run ", r)
        start = time.time()
        trackin_train_test_influences = tracin_utils.get_test_influence(batched_dataset, smiles_matrix_val, y_c_val, basic_models)
        end = time.time()
        print(datetime.timedelta(seconds=end - start))
        results.append(trackin_train_test_influences)
        
        
        #the influences are calculated as before, just with the TP-only validation set this time
        print("TracIn Test-Influence (only TP validation set) on scaled model in run ", r)
        start = time.time()
        trackin_train_test_influences_pos_only = tracin_utils.get_test_influence(batched_dataset, x_val_pos, y_c_val_pos, basic_models)
        end = time.time()
        print(datetime.timedelta(seconds=end - start))
        results_pos_only.append(trackin_train_test_influences_pos_only)
        
        
        print("TracIn Self-Influence on scaled model in run ", r)
        start = time.time()
        trackin_train_self_influences = tracin_utils.get_self_influence(batched_dataset, basic_models)
        end = time.time()
        print(datetime.timedelta(seconds=end - start))
        results_self.append(trackin_train_self_influences)
              
        tf.keras.backend.clear_session()
        rmtree(checkpoint_dir)
            
        
    
        importances = results[r].get("test_influences") * (-1)
        flags, flags_alt = process_ranking(y_p_train, importances)
        temp[r,0] = time.time() - start_all
        temp[r,1] = precision_score(y_f_train, flags_alt[idx_train])
    
        temp[r,2] = precision_score(y_c_train, flags[idx_train])
    
        #get EF10 for FPs and TPs
        temp[r,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
        temp[r,4] = enrichment_factor_score(y_c_train, flags[idx_train])
    
        #get BEDROC20 for FPs and TPs
        temp[r,5] = bedroc_score(y_f_train, importances[idx_train], reverse=False)
        temp[r,6] = bedroc_score(y_c_train, importances[idx_train], reverse=True)
    
        #get diversity for FPs
        idx_fp = np.where(flags_alt == 1)[0]
        mols_fp = [mols_train[x] for x in idx_fp]
        temp[r,7] = get_scaffold_rate(mols_fp)
    
        #get diversity for TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols_train[x] for x in idx_tp]
        temp[r,8] = get_scaffold_rate(mols_tp)
        
        #optionally fill up logger
        if log_predictions is True:
            logs_temp = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                       flags[idx_train], importances[idx_train], "TracIn_test")
          #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
          #i replicates
            logs = pd.concat([logs, logs_temp], axis=0, ignore_index=True)
     
     #performance is calculated as before, just with the influences generated via the TP-only validation set this time 

        importances = results_pos_only[r].get("test_influences") * (-1)
        flags, flags_alt = process_ranking(y_p_train, importances)
        
        temp_pos[r,0] = time.time() - start_all
        temp_pos[r,1] = precision_score(y_f_train, flags_alt[idx_train])
    
        temp_pos[r,2] = precision_score(y_c_train, flags[idx_train])
    
        #get EF10 for FPs and TPs
        temp_pos[r,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
        temp_pos[r,4] = enrichment_factor_score(y_c_train, flags[idx_train])
    
        #get BEDROC20 for FPs and TPs
        temp_pos[r,5] = bedroc_score(y_f_train, importances[idx_train], reverse=False)
        temp_pos[r,6] = bedroc_score(y_c_train, importances[idx_train], reverse=True)
    
        #get diversity for FPs
        idx_fp = np.where(flags_alt == 1)[0]
        mols_fp = [mols_train[x] for x in idx_fp]
        temp_pos[r,7] = get_scaffold_rate(mols_fp)
    
        #get diversity for TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols_train[x] for x in idx_tp]
        temp_pos[r,8] = get_scaffold_rate(mols_tp)
        
        #optionally fill up logger
        if log_predictions is True:
            logs_temp_pos = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                       flags[idx_train], importances[idx_train], "TracIn_test_pos")
          #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
          #i replicates
            logs_pos = pd.concat([logs_pos, logs_temp_pos], axis=0, ignore_index=True)
        
           
            
           

        importances = results_self[r].get("self_influences") * (-1)
        flags, flags_alt = process_ranking(y_p_train, importances)
        temp_self[r,0] = time.time() - start_all
        temp_self[r,1] = precision_score(y_f_train, flags_alt[idx_train])
    
        temp_self[r,2] = precision_score(y_c_train, flags[idx_train])
    
        #get EF10 for FPs and TPs
        temp_self[r,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
        temp_self[r,4] = enrichment_factor_score(y_c_train, flags[idx_train])
    
        #get BEDROC20 for FPs and TPs
        temp_self[r,5] = bedroc_score(y_f_train, importances[idx_train], reverse=False)
        temp_self[r,6] = bedroc_score(y_c_train, importances[idx_train], reverse=True)
    
        #get diversity for FPs
        idx_fp = np.where(flags_alt == 1)[0]
        mols_fp = [mols_train[x] for x in idx_fp]
        temp_self[r,7] = get_scaffold_rate(mols_fp)
    
        #get diversity for TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols_train[x] for x in idx_tp]
        temp_self[r,8] = get_scaffold_rate(mols_tp)
        
        #optionally fill up logger
        if log_predictions is True:
            logs_temp_self = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                       flags[idx_train], importances[idx_train], "TracIn_self")
          #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
          #i replicates
            logs_self = pd.concat([logs_self, logs_temp_self], axis=0, ignore_index=True)
        gc.collect()
    #results of both approaches are returned   
    return temp, temp_pos, temp_self, logs, logs_pos, logs_self


def run_tracin_importance(
        db_train: pd.DataFrame,
        db_val: pd.DataFrame,
        name: str,
        representation: str = "ecfp",
        replicates: int = 5,
        log_predictions: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    
    """Executes TracIn analysis on given dataset
   3 influence scores are computed: 
     - The first uses the self_influence definition,
       calculatiung the influence of a training sample on its own prediction by 
       squaring its gradients.
     - The second uses the test_influence definition,
       calculating the influence of a training sample on the prediction of the 
       test samples by multiplying its gradients with the gradients of all test
       samples, summing over the weights and averaging over the test samples.
     - The third method works the same as the second, but only using true positives
       as as test samples 
   
    
   The training samples are then ranked according to their influences, or "importances". Interestingly, the labels first have to be flipped
   (by multiplying with -1) to be of the same orientation as the other importance score methods. Top 10% are 
   defined as proponents, bottom 10% as opponents. Finally, the 
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
        Tuple containing two arrays (1,9) with precision@90 for FP and TP retrieval, 
        scaffold diversity and training time, and two dataframes (V,5) with
        SMILES, true labels and raw predictions (logs); one of each for the normal TracIn implementation, one for the
        implementation only using TPs as the validation  set
    
    
    """
    #calculate mols for structure similarity analysis
    mols_train = list(db_train["SMILES"])
    mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
    mols_val = list(db_val["SMILES"])
    mols_val = [Chem.MolFromSmiles(x) for x in mols_val]
    
    #loading in the molecular representations according to the "representation
    #argument; molecular representations were calculated and stored via the 
    #"descr_export_pipeline_jh.py
    train_path = "../../Datasets_descr/" + name + "/" + name + "_" + representation + "_train.pkl"
    x_train = pd.read_pickle(train_path)
    x_train = x_train.to_numpy()
    val_path = "../../Datasets_descr/" + name + "/" + name + "_" + representation + "_val.pkl"
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
        assert x_val.shape ==(len(y_p_val), 208), "x_train has wrong shape"
    elif(representation == "ecfp"):
        assert x_train.shape ==(len(y_p_train), 1024), "x_train has wrong shape"
        assert x_val.shape ==(len(y_p_val), 1024), "x_train has wrong shape"

    

    #The indices of all true positives from the validation set are extracted and
    #used to generate a TP-only validation set
    idx_pos = []
    for i in range(len(y_c_val)):
        if y_c_val[i]==1:
            idx_pos.append(i)
        
    x_val_pos = x_val[idx_pos]
    y_c_val_pos = y_c_val[idx_pos]
        


    #renaming for ease of reading
    train_labels = y_p_train
    
    #creating a dataset out of the training set to feed
    #allow batchwhise influence calculation
    dataset = tf.data.Dataset.from_tensor_slices((x_train, train_labels))
    
    #split dataset into batches
    batched_dataset = dataset.batch(batch_size=512)
    
    
    
    #for frequency of saving checkpoints:
# =============================================================================
#     BATCH_SIZE = 512
#     STEPS_PER_EPOCH = train_labels.size / BATCH_SIZE
#     SAVE_PERIOD = 30
#     EPOCHS = 100
# =============================================================================

    #initialize result containers
    temp = np.zeros((replicates,9))
    temp_pos = np.zeros((replicates,9))
    temp_self = np.zeros((replicates,9))
    logs = pd.DataFrame([])
    logs_temp = pd.DataFrame([])
    logs_pos = pd.DataFrame([])
    logs_temp_pos = pd.DataFrame([])
    logs_self = pd.DataFrame([])
    logs_temp_self = pd.DataFrame([])
    
    results = []
    results_pos_only = []
    results_self = []

    
    
    for r in range(replicates):
      
        start_time1 = time.time()
        checkpoint_path = "test_influence" + str(r) +"_" + name +"_/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
    
        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq= "epoch")
    
        # Create a new model instance
        basic_model = tracin_utils.make_model(x_train = x_train, random_state=r)
        print("fitting basic model in run ", r)
        # Save the weights using the `checkpoint_path` format
        #had to take the save at epoch 0 step out for model 7
        basic_model.save_weights(checkpoint_path.format(epoch=0))
        start = time.time()
        # Train the model with the new callback
        basic_model.fit(x_train, 
                  train_labels,
                  epochs=100, 
                  batch_size=512, 
                  callbacks=[cp_callback],
                  validation_split = 0.1,
                  verbose=0)
        end = time.time() - start
        print("Fitting time: ", end)
        start = time.time()
    
        basic_models = []
        for i in [30, 60, 90]:
            basic_model = tracin_utils.make_model(x_train = x_train, random_state=r)
            basic_model.load_weights(checkpoint_path.format(epoch=i)).expect_partial()
            basic_models.append(basic_model)
        end = time.time()
        print(datetime.timedelta(seconds=end - start))  
    
        end_time1 = time.time() -start_time1
        
        tracin_time =time.time()
        print("TracIn Test-Influence on scaled model in run ", r)
        start = time.time()
        trackin_train_test_influences = tracin_utils.get_test_influence(batched_dataset, x_val, y_c_val, basic_models)
        end = time.time()
        print(datetime.timedelta(seconds=end - start))
        results.append(trackin_train_test_influences)
        tracin_time_end = time.time() - tracin_time
        
        
        tracin_pos_time =time.time()
        #the influences are calculated as before, just with the TP-only validation set this time
        print("TracIn Test-Influence (only TP validation set) on scaled model in run ", r)
        start = time.time()
        trackin_train_test_influences_pos_only = tracin_utils.get_test_influence(batched_dataset, x_val_pos, y_c_val_pos, basic_models)
        end = time.time()
        print(datetime.timedelta(seconds=end - start))
        results_pos_only.append(trackin_train_test_influences_pos_only)
        tracin_pos_time_end = time.time() - tracin_pos_time
        
        
        tracin_self_time =time.time()
        print("TracIn Self-Influence on scaled model in run ", r)
        start = time.time()
        trackin_train_self_influences = tracin_utils.get_self_influence(batched_dataset, basic_models)
        end = time.time()
        print(datetime.timedelta(seconds=end - start))
        results_self.append(trackin_train_self_influences)
        tracin_self_time_end = time.time() - tracin_self_time      
        
        
        tf.keras.backend.clear_session()
        rmtree(checkpoint_dir)
        
    
        importances = results[r].get("test_influences") * (-1)
        flags, flags_alt = process_ranking(train_labels, importances)
        temp[r,0] = tracin_time_end + end_time1
        temp[r,1] = precision_score(y_f_train, flags_alt[idx_train])
    
        temp[r,2] = precision_score(y_c_train, flags[idx_train])
    
        #get EF10 for FPs and TPs
        temp[r,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
        temp[r,4] = enrichment_factor_score(y_c_train, flags[idx_train])
    
        #get BEDROC20 for FPs and TPs
        temp[r,5] = bedroc_score(y_f_train, importances[idx_train], reverse=False)
        temp[r,6] = bedroc_score(y_c_train, importances[idx_train], reverse=True)
    
        #get diversity for FPs
        idx_fp = np.where(flags_alt == 1)[0]
        mols_fp = [mols_train[x] for x in idx_fp]
        temp[r,7] = get_scaffold_rate(mols_fp)
    
        #get diversity for TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols_train[x] for x in idx_tp]
        temp[r,8] = get_scaffold_rate(mols_tp)
        
        #optionally fill up logger
        if log_predictions is True:
            logs_temp = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                       flags[idx_train], importances[idx_train], "TracIn_test")
          #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
          #i replicates
            logs = pd.concat([logs, logs_temp], axis=0, ignore_index=True)
     
     #performance is calculated as before, rust with the influences generated via the TP-only validation set this time 
    
        importances = results_pos_only[r].get("test_influences") * (-1)
        flags, flags_alt = process_ranking(train_labels, importances)
        
        temp_pos[r,0] = tracin_pos_time_end + end_time1
        temp_pos[r,1] = precision_score(y_f_train, flags_alt[idx_train])
    
        temp_pos[r,2] = precision_score(y_c_train, flags[idx_train])
    
        #get EF10 for FPs and TPs
        temp_pos[r,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
        temp_pos[r,4] = enrichment_factor_score(y_c_train, flags[idx_train])
    
        #get BEDROC20 for FPs and TPs
        temp_pos[r,5] = bedroc_score(y_f_train, importances[idx_train], reverse=False)
        temp_pos[r,6] = bedroc_score(y_c_train, importances[idx_train], reverse=True)
    
        #get diversity for FPs
        idx_fp = np.where(flags_alt == 1)[0]
        mols_fp = [mols_train[x] for x in idx_fp]
        temp_pos[r,7] = get_scaffold_rate(mols_fp)
    
        #get diversity for TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols_train[x] for x in idx_tp]
        temp_pos[r,8] = get_scaffold_rate(mols_tp)
        
        #optionally fill up logger
        if log_predictions is True:
            logs_temp_pos = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                       flags[idx_train], importances[idx_train], "TracIn_test_pos")
          #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
          #i replicates
            logs_pos = pd.concat([logs_pos, logs_temp_pos], axis=0, ignore_index=True)
        
           
            
           

        importances = results_self[r].get("self_influences") * (-1)
        flags, flags_alt = process_ranking(train_labels, importances)
        temp_self[r,0] = tracin_self_time_end + end_time1
        temp_self[r,1] = precision_score(y_f_train, flags_alt[idx_train])
    
        temp_self[r,2] = precision_score(y_c_train, flags[idx_train])
    
        #get EF10 for FPs and TPs
        temp_self[r,3] = enrichment_factor_score(y_f_train, flags_alt[idx_train])
        temp_self[r,4] = enrichment_factor_score(y_c_train, flags[idx_train])
    
        #get BEDROC20 for FPs and TPs
        temp_self[r,5] = bedroc_score(y_f_train, importances[idx_train], reverse=False)
        temp_self[r,6] = bedroc_score(y_c_train, importances[idx_train], reverse=True)
    
        #get diversity for FPs
        idx_fp = np.where(flags_alt == 1)[0]
        mols_fp = [mols_train[x] for x in idx_fp]
        temp_self[r,7] = get_scaffold_rate(mols_fp)
    
        #get diversity for TPs
        idx_tp = np.where(flags == 1)[0]
        mols_tp = [mols_train[x] for x in idx_tp]
        temp_self[r,8] = get_scaffold_rate(mols_tp)
        
        #optionally fill up logger
        if log_predictions is True:
            logs_temp_self = run_logger(mols_train, idx_train, y_f_train, y_c_train, flags_alt[idx_train],
                                       flags[idx_train], importances[idx_train], "TracIn_self")
          #since logs_temp only catch the samples of one iteration, logs is a concatenation of all 
          #i replicates
            logs_self = pd.concat([logs_self, logs_temp_self], axis=0, ignore_index=True)
        gc.collect()
    #results of both approaches are returned   
    return temp, temp_pos, temp_self, logs, logs_pos, logs_self
