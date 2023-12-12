#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:54:31 2023

@author: joshua
"""
import warnings
from utils_jh import *
import numpy as np
from rdkit import Chem
import pandas as pd
import argparse
import os
import time
import gc
"""
This pipeline allows benchmarking the detection of False positive samples in 
High Throughput screen data. To use this pipeline, one should use either the 
datasets used in this project, or setup an equivalent data setup, having 
a train and a val set, as well as primary data for all samples and confirmatory 
data for some primary actives. 

The pipeline can be used to compare different methods. The classical Score and 
Fragment filter methods represent the currently methods used in HTS pipelines. 
New methods developed here are based on different data valuation approaches, 
calculating training sample importance/influence values and using these values 
to predict false positives.

The pipeline allows the FP prediction with the following methods:
    
    - Score (using the primary screen activity score for prediction)
    
    - Fragment-Filter (using the RDKIT Fragment Filter (PAINS) prediction)
    
    - Data-Shapley / KNN (using Data shapley to calculate sample importance using
                          a separate test set for prediction performance measurement.
                          These importance values are then used for FP prediction)
    
    - TracIn: (Tracing gradient descent in deep neural networks)
    
                Test-Influence (using gradients on separate testset to predict training 
                                sample influence on said test set. These influence values 
                                are then used to predict FPs)
                
                Positive-test-influence (same as above, but only using actives in the 
                                         separate test set)
                
                Self-importance (Train samples gradients on their own prediction are used
                                 to calculate their influence instead of using a separate
                                 test set)
        
    - Catboost: (using Catboosts built-in get_object_importance function)
    
                Test-Influence (Using a separate test set for the built-in importance function
                                together with the entire training set for importance prediction
                                and using the resulting importance values to predict FPs)
                
                Self-Influence (Using all primary actives for the built-in importance function
                                together with the entire training set)
        
    - DVRL (Data Valuation using Reinforcement Learning - uses a reinforcement
            learning approach for influence prediction, where performance on a 
            separate test set is used for reinforcement learning. The resulting
            importance scores are used for FP prediction)
       
    - MVS-A (MultiVariance Sampling Analysis: uses the effects samples have on
             LightGBM classifiers to calculate their influence scores. These are
             then used for FP prediction) 
    
Unfortunately, some data valuation approaches have contradicting package requirements.
Therefore, the pipeline allows setting of an "environment" parameter, which will
import the correct utility files based on the methods used. However, one also 
has to call the pipeline in the according virtual environment!

Environment: TracIn     --> methods: Score, Fragment_Filter, TracIn, Catboost, MVS-A
             
             dvrl       --> methods: Score, Fragment_Filter, DVRL
             
             shapley    --> methods: Score, Fragment_Filter, Data Shapley
        
"""


# =============================================================================
parser = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default="all",
                    help="Which dataset from ../Datasets to use for the analysis, options: [all, specific_name]")

parser.add_argument('--knn', default="yes",
                    help="Whether to use KNN for the run, options: [yes, no]")

parser.add_argument('--dvrl', default="yes",
                    help="Whether to use DVRL for the run, options: [yes, no]")

parser.add_argument('--tracin', default="yes",
                    help="Whether to use TracIn for the run, options: [yes, no]")

parser.add_argument('--mvs_a', default="yes",
                    help="Whether to use MVS-A for the run, options: [yes, no]")

parser.add_argument('--catboost', default="yes",
                    help="Whether to use CatBoost for the run, options: [yes, no]")

parser.add_argument('--score', default="yes",
                    help="Whether to use assay readouts for the run, options: [yes, no]")

parser.add_argument('--fragment_filter', default="yes",
                    help="Whether to use a fragment filter for the run, options: [yes, no]")

parser.add_argument('--filter_type', default="PAINS",
                    help="Which fragment set to use for the run, options: [PAINS, PAINS_A, PAINS_B, PAINS_C, NIH]")

parser.add_argument('--representation', default="ECFP",
                    help="Which sample representation to use for the run, options: [ECFP, SMILES, RDKIT]")

parser.add_argument('--replicates', default=5, type=int,
                    help="How many replicates to use for the influence methods")

parser.add_argument('--filename', default="output",
                    help="Name to use when saving performance results")

parser.add_argument('--log_predictions', default="yes",
                    help="Whether to log raw predictions, options: [yes, no]")

parser.add_argument('--environment', default="others",
                    help="What virtual environment is being used: [dvrl, shapley, others]")

args = parser.parse_args()


#conditional utility imports to avoid contradicting imports crashing    
if args.environment.lower() == "others":
    try:
        from influence_mvs_a import *
        import tensorflow as tf
        from influence_tracin import *
        from run_score import *
        from run_filter import *
        from influence_catboost import *
    except ImportError:
        warnings.warn('TracIn evals_tracin failed to import', ImportWarning)

elif args.environment.lower() == "dvrl":
    try:
      from influence_dvrl import run_dvrl
      from run_score import *
      from run_filter import *
    except ImportError:
        warnings.warn('Shapey evals_shapley failed to import', ImportWarning)
        
elif args.environment.lower() == "shapley":
    try:
      from influence_shapley import *
      from run_score import *
      from run_filter import *
    except ImportError:
        warnings.warn('DVRL evals_dvrl failed to import', ImportWarning)
###############################################################################

def main(dataset,
         knn,
         score,
         dvrl,
         fragment_filter,
         catboost,
         mvs_a,
         tracin,
         filter_type,
         representation,
         replicates,
         filename,
         log_predictions,
         environment):
    
    #setting the GPU for TracIn
    if environment == "others":
        
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_visible_devices(physical_devices[0], "GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    

    #turn log_predictions into boolean, then check if it is compatible with other params
    if log_predictions == "no":
        log_predictions = False
    else:
        #if logging is true, create a new folder in the Logs/eval folder called filename to 
        #save all logs of this runthrough
        log_predictions = True
        if not(os.path.isdir("../../Logs/eval/" + filename)):
               os.mkdir("../../Logs/eval/" + filename)
  
          

    
    #set datasets to be analyzed, either all or one specific
    if dataset != "all":
        dataset_names = [dataset]
    else:
        dataset_names = os.listdir("../../Datasets")    
        dataset_names = [x for x in dataset_names]
        
        
    #initialize result boxes for all possible algorithms 
    #all influence methods also export the data for all replicates
    #Score and Fragment_Filter do not, as replicates do not make sense here
    
    column_number = 20 + 9 * replicates
    mvs_a_box = np.zeros((len(dataset_names), column_number))
    catboost_self_box = np.zeros((len(dataset_names), column_number))
    catboost_test_box = np.zeros((len(dataset_names), column_number))
    filter_box = np.zeros((len(dataset_names), 20))  
    knn_box= np.zeros((len(dataset_names), column_number))
    score_box = np.zeros((len(dataset_names), 20))
    dvrl_box = np.zeros((len(dataset_names), column_number))
    tracin_box = np.zeros((len(dataset_names), column_number))
    tracin_box_pos = np.zeros((len(dataset_names), column_number))
    tracin_box_self = np.zeros((len(dataset_names), column_number))
    print("[eval]: Beginning eval run...")
    print("[eval]: Run parameters:")
    print(f"        dataset: {dataset_names}")
    print(f"        environment: {environment}")
    print(f"        algorithms: knn = {knn}, mvs_a = {mvs_a}, catboost = {catboost}, dvrl = {dvrl}, tracin = {tracin}, score = {score}, fragment_filter: {fragment_filter}")
    print(f"        representation: {representation}")
    print(f"        replicates: {replicates}")
    print(f"        prediction logging: {log_predictions}")
    print(f"        file identifier: {filename}")
    
    #loop analysis over all datasets
    for i in range(len(dataset_names)):

        #load i-th dataset, get mols and then representation for training and validation set seperately
        print("----------------------")
        name = dataset_names[i]
        print(f"[eval]: Processing dataset: {name}")
        #the smiles and activity labels are imported from the datasets folder
        #the files are stored in dataframes (df)
        df_train = pd.read_csv("../../Datasets/" + name + "/" + name + "_train.csv")
        df_val = pd.read_csv("../../Datasets/" + name + "/" + name + "_val.csv")
        
        mols_train = list(df_train["SMILES"])
        mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
        
        mols_val = list(df_val["SMILES"])
        mols_val = [Chem.MolFromSmiles(x) for x in mols_val]
        
        #the features (x_train/val), either rdkit or ecfp, were precalculated for consistancy
        #and speed, and are imported from the Datasets_descr folder
        if representation != "smiles":
            train_path = "../../Datasets_descr/" + name + "/" + name + "_" + representation + "_train.pkl"
            val_path = "../../Datasets_descr/" + name + "/" + name + "_" + representation + "_val.pkl"
        
        
            x_train = pd.read_pickle(train_path)
            x_train = x_train.to_numpy()
        
            x_val = pd.read_pickle(val_path)
            x_val = x_val.to_numpy()

        #get labels for the analysis and get random-guess probabilities for
        #FP and TP. y_p_train are the labels from the primary screen for the training set,
        #y_c_train and y_f_train are the TP/FP labels for the primary actives of the training set
        #which also had a readout in the confirmatory. idx_train is the position of these compounds
        #in the primary screen of the training set
        y_p_train, y_c_train, y_f_train, idx_train = get_labels(df_train)
        fp_rate_train = np.sum(y_f_train) / len(y_f_train)
        tp_rate_train = 1 - fp_rate_train
        
        #for the scoring method to work, additional information of the validation set is also required
        #here, only y_c_val is really necessary
        y_p_val, y_c_val, y_f_val, idx_val = get_labels_val(df_val)
        
        
        #logs are collected over the course of the pipeline and finally 
        #all merged to one log
        log_list = []
        
        
        #every method is checked for its corresponding flag
        #if set, the fps are predicted, all scoring values for all replicates
        #are saved and stored in boxes
        if knn == "yes":
            print("[eval]: Running KNN analysis...")
            knn_results, knn_log = run_knn(df_train, df_val, dataset_names[i], representation, 
                                          replicates, log_predictions)
            knn_box = store_row(knn_box, knn_results, replicates,fp_rate_train, tp_rate_train, i)
            log_list.append(knn_log)
            print("[eval]: KNN analysis finished")     
       
        if mvs_a == "yes":
            print("[eval]: Running MVS-A analysis...")
            mvsa_results, mvsa_log = run_mvsa(mols_train, x_train, y_p_train, y_f_train, y_c_train, idx_train, replicates,
                                     log_predictions)
            mvs_a_box = store_row(mvs_a_box, mvsa_results, replicates, fp_rate_train, tp_rate_train, i)
            log_list.append(mvsa_log)
            print("[eval]: MVS-A analysis finished")

        if catboost == "yes":
            print("[eval]: Running CatBoost analysis...")
            cb_self_results, cb_test_results, cb_log_self, cb_log_test = run_catboost(df_train, df_val, name,replicates= replicates,
                                       log_predictions= log_predictions)
            catboost_self_box = store_row(catboost_self_box, cb_self_results,replicates, fp_rate_train, tp_rate_train, i)
            catboost_test_box = store_row(catboost_test_box, cb_test_results,replicates, fp_rate_train, tp_rate_train, i)
            log_list.append(cb_log_self)
            log_list.append(cb_log_test)
            print("[eval]: CatBoost analysis finished") 
            
        if dvrl == "yes":
            print("[eval]: Running DVRL analysis...")
            dvrl_results, dvrl_log = run_dvrl(df_train, df_val, dataset_names[i], representation, 
                                          replicates, log_predictions)
            dvrl_box = store_row(dvrl_box, dvrl_results, replicates,fp_rate_train, tp_rate_train, i)
            log_list.append(dvrl_log)
            print("[eval]: DVRL analysis finished") 
        
        
        #tracin has the additional representation possibility of smiles, using
        #a long short term memory model, instead of the classic DNN used for 
        #tabular data
        if tracin == "yes":
            print("[eval]: Running TracIn analysis...")
            if (representation == "smiles"):
                print("[eval]: Running TracIn on LSTM model")
                tracin_results, tracin_pos_results, tracin_self_results, tracin_log, tracin_pos_log, tracin_self_log = run_tracin_importance_LSTM(df_train, df_val, name,  
                                              replicates, log_predictions)
            else:   
                tracin_results, tracin_pos_results, tracin_self_results, tracin_log, tracin_pos_log, tracin_self_log = run_tracin_importance(df_train, df_val, name, representation, 
                                              replicates, log_predictions)
                
            tracin_box = store_row(tracin_box, tracin_results,replicates, fp_rate_train, tp_rate_train, i)
            tracin_box_pos = store_row(tracin_box_pos, tracin_pos_results,replicates,fp_rate_train, tp_rate_train, i)
            tracin_box_self = store_row(tracin_box_self, tracin_self_results,replicates, fp_rate_train, tp_rate_train, i)
            log_list.append(tracin_log)
            log_list.append(tracin_pos_log)
            log_list.append(tracin_self_log)
            print("[eval]: TracIn analysis finished")    
        
        #for score and filter, the score log and filter log are repeated 5 times,
        #as no replicates are calculated here - so in order to be merged with the
        #other methods via an inner join, the number of samples need to match
        if score == "yes":
            print("[eval]: Running score analysis...")
            score_results, score_log = run_score(df_train, mols_train, idx_train, y_p_train,
                                         y_f_train, y_c_train, log_predictions)
            score_box = store_row_without_replicates(score_box, score_results, fp_rate_train, tp_rate_train, i)  
            
            #as score is a direct sorting after activity, replicates do not make sense.
            #however, for consistency all functions need the same number of logs,
            #therefore the log is just repeated rep times
            score_log_list = []
            for rep in range(replicates):
                score_log_list.append(score_log)
            score_log = pd.concat(score_log_list, axis=0, ignore_index=True)
            log_list.append(score_log)
            print("[eval]: Score analysis finished")
        
        if fragment_filter == "yes":
            print("[eval]: Running filter analysis...")
            fragment_filter_results, filter_log = run_filter(mols_train, idx_train, filter_type, y_f_train, y_c_train,
                                        log_predictions)
            filter_box = store_row_without_replicates(filter_box, fragment_filter_results, fp_rate_train, tp_rate_train, i)
            
            #as with results, this method does not need replicates, but for logging
            #consistency the resulting logs are repeated rep times.
            filter_log_list = []
            for rep in range(replicates):
                filter_log_list.append(filter_log)
            filter_log = pd.concat(filter_log_list, axis=0, ignore_index=True)
            log_list.append(filter_log)
            print("[eval]: Filter analysis finished")
        


        #optionally store logs (raw predictions for all compounds)
        if log_predictions is True:
            for i in range(len(log_list)):
                if(i == 0):
                    logs = log_list[i]
                else:
                    df2_last_three_columns = log_list[i].iloc[:, -3:]
                    logs = pd.concat([logs, df2_last_three_columns], axis=1)
 
            logpath = "../../Logs/eval/" + filename + "/" + filename + "_" + name + "_" + representation + ".csv"
            
            logs.to_csv(logpath)
            print(f"[eval]: Log saved at {logpath}")
        gc.collect()
            
    #save results for all algorithms as .csv files
    #the influence methods are being saved separately from score and fragment_filter,
    #as all replicates are saved, needing a special saving function
    save_results(
        [mvs_a_box, catboost_self_box, catboost_test_box, knn_box, dvrl_box, tracin_box, tracin_box_pos, tracin_box_self],
        dataset_names,
        representation,
        replicates,
        filename,
        filter_type
        )
    
    save_results_without_replicates(
        [score_box, filter_box],
        dataset_names,
        representation,
        filename,
        filter_type
        )
    print("----------------------")
    print("[eval]: Results saved in ../../Results/" + filename + "/*_" + filename + "_" + representation + ".csv")
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main(dataset = args.dataset,
         knn = args.knn.lower(),
         score = args.score.lower(),
         mvs_a = args.mvs_a.lower(),
         catboost = args.catboost.lower(),
         fragment_filter = args.fragment_filter.lower(),
         dvrl = args.dvrl.lower(),
         tracin = args.tracin.lower(),
         filter_type = args.filter_type.upper(),
         representation = args.representation.lower(),
         replicates = args.replicates,
         filename = args.filename,
         log_predictions = args.log_predictions,
         environment = args.environment)
