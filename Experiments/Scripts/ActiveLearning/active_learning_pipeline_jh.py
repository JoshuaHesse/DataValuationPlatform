#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:42:43 2023

@author: joshua

This pipeline should be used to calculate the active learing performance on 
one or multiple datasets. It can be used to calculate classical, here referred 
to as "greedy" active learning. In this approach, a samll subset of samples is
tested empirically, resulting in their acitvity labels. These are used to train
a classifier that predicts the activity of the remaining samples of the given
dataset. Then, the samples with the highest probabilities to be actives are chosen,
validated, and added to the training pool. This procedure can be iterated 
multiple times and the number of actives in the training set can be compared to the
random sampling. If this approach works well, active samples should accumulate
quickly in the training set, much quicker than in the random sampling control.

This pipeline can further be used to perform an alternative active learning 
approach that is based on training sample influence calculations, here referred 
to as "influence active learning". In this apporach a small subset of 
samples is tested empirically, resulting in their acitvity labels. These labels 
are used to train a machine learning model. However, instead of using the
model to predict the remaining samples' activity, the model is analyzed regarding
the influence each training sample had on its construction. These influence
scores are then used to predict influence scores for the remaining samples using
regression.  Then, the samples with the highest predicted influence scores are picked,
validated, and added to the training pool. This procedure can be iterated 
multiple times and the number of actives in the training set can be compared to the
random sampling and the greedy approach. If this approach works well, 
active samples should accumulate quickly in the training set, much quicker than
in the random sampling control.

When running the script from the terminal, the following flags can be set:
   
    dataset: if set, a specific dataset can be given. Otherwhise, all datasets
             in the Datasets folder will be analyzed
             
    representation: if set, ECFP or RDKIT can be set determine the type of 
                    molecular represetation
    
    influence:  MVSA, TracIn, TracIn_pos, TracIn_self, DVRL, Shapley,
                Catboost_self, or Catboost_test can be set to determine 
                the Influence calculation function.
                
                CAVEAT: When using Shapley, the code must be run in the datascope
                        environment rather than the standard one. Tensorflow 
                        functionality will not be accessible, meaning e.g. the 
                        DNN regression is not usable.
               
    regression: Can either be set to "all", comparing all regression methods, or 
                to a specific Regression function. This function will be trained
                on the few initial training samples' influence scores to predict
                influence scores of the remaining untested samples
              
    sampling:   Can either be set to "all", comparing both sampling methods, or 
                to a specific function. The sampling methods differ in how the
                next iteration of training samples is picked. "Greedy" picks
                the samples with the highest mean predicted scores, while "ucb", 
                upper confidence bound, takes into account the variance of that
                prediction as well.
            
    steps:      Can be set to define the number of iterations of active learning
                to be done.
    
    step_size:  Can be set to define the fraction (in percent of the original 
                dataset size) of the entire dataset that should be picked 
                during each iteration.
                
    random_size: Can be set to define the fraction (in percent of the original
                 dataset size) to be used as the starting training.
    
    replicates: Can be set to define the number of replicates for each dataset
                and each method.
                
    fielname: If set, the input will be used as the name for the folder where
              the results will be saved

Steps:
    1. Load dataset and load ECFPs / RDKIT descr of train and val set
        ->  the val set is used for some influence functions that require a 
            separate val set. For influence functions that use self-influence on
            the training set, the val set will not be used
    2. Create labels from the dataset csv files for train and val
    3. Run all combinations of Influence-, regression-, and sampling strategies
    4. Run greedy active learning
    5. Run random sampling
    6. Print results to .png
    7. Save results as .csv file

for further detail regarding the functionality of the pipeline, read 
actives_found.py
"""

import argparse
import random
import os
import warnings
import pandas as pd
from utils_jh import get_labels
import time
from actives_found import actives_found, greedy_actives_found, random_actives_found
from data_analysis import plot_learning, save_data, plot_learning_bands



# =============================================================================
parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default="all",
                    help="Which dataset from ../Datasets to use for the analysis, options: [all, specific_name]")

parser.add_argument('--representation', default="ECFP",
                    help="Which sample representation to use for the run, options: [ECFP, RDKIT]")

parser.add_argument('--influence', default="MVSA",
                    help="Which sample influence function to use for the run, options: [MVSA, TracIn, TracIn_pos, TracIn_self, Catboost_self, Catboost_test, DVRL, Shapley]")

parser.add_argument('--regression', default="all",
                    help="Which regression function to use for the run, options: [all, GPR, GPR_gpflow, LGBM, SVR, DNN]")

parser.add_argument('--sampling', default="all",
                    help="Which sampling function to use for the run, options: [all,greedy, ucb]")

parser.add_argument('--steps', default=5, type = int,
                    help="How many iterations of active learning to do")

parser.add_argument('--step_size', default=1, type = float,
                    help="Which size the steps should have (in % of the entire dataset)")

parser.add_argument('--random_size', default=1, type = float,
                    help="Which size the starting fraction should have (in % of the entire dataset)")

parser.add_argument('--replicates', default=5, type = int,
                    help="Number of replicates")

parser.add_argument('--filename', default="output",
                    help="Name to use when saving performance results")



args = parser.parse_args()

try:
    if args.influence.lower() != "dvrl":
        import tensorflow as tf
    else:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
except ImportError:
    warnings.warn('Tensorflow failed to import', ImportWarning)

def main(dataset,
         representation,
         influence,
         regression,
         sampling,
         steps,
         step_size,
         random_size,
         replicates,
         filename):
    #GPU setting. This has to be done in the tracin_utils.py file as well,
    #setting the same GPU (e.i. both 0 or both 1)!
    #when in the datascope environment, tensorflow is not available -> Error
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_visible_devices(physical_devices[0], "GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except NameError:
        warnings.warn('GPU failed to set')
        
    #if a specific dataset is given in the args, it is set here. Otherwhise,
    #all datasets in the datasets folder are used
    if dataset != "all":
        dataset_names = [dataset]
    else:
        dataset_names = os.listdir("../Datasets")    
        dataset_names = [x for x in dataset_names]
        
   
    
    #in the results folder a new folder in created with the filename.
    #this folder will be filled with 1 png and 2 csv files per dataset    
    if not(os.path.isdir("../Results/" + filename)):
           os.mkdir("../Results/" + filename)
    
        
    #influence function according to args
    influence_functions = [influence]
    #influence_functions = ["mvsa", "catboost_self", "catboost_test", "tracin", "tracin_pos", "tracin_self"]
    
    #regression function according to args
    if regression != "all":
        regression_functions = [regression]
    else:
        regression_functions = ["GPR", "GPR_gpflow", "LGBM", "SVR", "DNN"]
        
        
    
    #sampling function according to args
    if sampling != "all":
        sampling_functions = [sampling]
    else:
        sampling_functions = ["greedy", "ucb"]

     
    
    print("[eval]: Beginning eval run...")
    print("[eval]: Run parameters:")
    print(f"        dataset: {dataset_names}")
    print(f"        representation: {representation}")
    print(f"        Influence Function: {influence_functions}")
    print(f"        regression functions: {regression_functions}")
    print(f"        sampling strategies: {sampling_functions}")
    print(f"        Setup: steps = {steps}, stepsize = {step_size}, randomsize = {random_size}")
    print(f"        file identifier: {filename}")

    #active learning calculations and results savings are done for each dataset
    #individually
    for i in range(len(dataset_names)): 
        print("----------------------")
        #current dataset 
        name = dataset_names[i]
        print(f"[eval]: Processing dataset: {name}")
        
        #a new folder for this dataset is created in the filename folder 
        if not(os.path.isdir("../Results/" + filename + "/" + name)):
               os.mkdir("../Results/" + filename + "/" + name)
        
       
        
        #the precalculated ECFP/RDKIT representations for the training set
        #are loaded from the Datasets_descr folder
        train_path = "../Datasets_descr/" + name + "/" + name + "_" + representation + "_train.pkl"
        x_train = pd.read_pickle(train_path)
        x_train = x_train.to_numpy()
        
        #labels are taken from the Datasets csv file
        train_df = pd.read_csv("../Datasets/" + name + "/" + name + "_train.csv")
        y_p_train, y_c_train, y_f_train, idx_train = get_labels(train_df)
        
        
        #the precalculated ECFP/RDKIT representations for the validation set
        #are loaded from the Datasets_descr folder
        val_path = "../Datasets_descr/" + name + "/" + name + "_" + representation + "_val.pkl"
        x_val = pd.read_pickle(val_path)
        x_val = x_val.to_numpy()
        
        #labels are taken from the Datasets csv file
        val_df = pd.read_csv("../Datasets/" + name + "/" + name + "_val.csv")
        y_p_val, y_c_val, y_f_val, idx_val = get_labels(val_df)
    
        
        
            
            
        #for each Dataset, all possible combinations of influence_function,
        #regression_function, and sampling_strategy are calculated 
        
        #combined names will hold strings of describing each combination of 
        #influence-, regression-, and sampling function
        combined_names = [] 
        
        #influence_learning_results will hold the results for each combination
        influence_learning_results = []
        
        #inlfuence_learning_times will hold the computational time taken
        #for each combination
        influence_learning_times = []
        
        #for each replicate, a different random seed is used to sample the 
        #initial training set that the active learning process is built on
        seeds = [random.randint(0, 9999) for _ in range(replicates)]
        for influence_calc in influence_functions:
            for regression_calc in regression_functions:
                for sampling_calc in sampling_functions:
                    
                    #results replicates holds the results of all replicates for
                    #this combination of functions
                    results_replicates = []
                    
                    #times replicates holds the computational time taken by each
                    #replicate
                    times_replicates = []
                    print(f"[eval]: Processing dataset: {name} via {influence_calc}, {regression_calc}, {sampling_calc}")
                    combined_name = influence_calc + "_" + regression_calc + "_" + sampling_calc
                    combined_names.append(combined_name)
                    
                    for rep in range(replicates):
                        print("[eval]: Replicate ", rep)
                       
                        beginning_time = time.time()
                        #each combination of influence-, regression-, and 
                        #sampling-strategy is saved as a string to label the data
                        #later on
                        #both these labels and the corresponding data are appended
                        #to the combined_names and influence_learning_results list
                        #--> label and data at same index
                        
                        #the number of actives found for each step are calculated
                        result_actives_found = actives_found(x_train = x_train,y_train = y_p_train, x_val=x_val, y_val = y_c_val, representation = representation, seed = seeds[rep], steps = steps, step_size = step_size, 
                                          random_size = random_size, influence_calc = influence_calc, regression_calc = regression_calc, sampling_strategy = sampling_calc)
                        
                        #for ease of comparison between datasets, the number of
                        #actives found is converted into a percentage of all actives
                        #present in the dataset
                        result_actives_found = result_actives_found/y_p_train.sum()
                    
                        end_time = time.time() - beginning_time
                        #replicate results and time are saved
                        times_replicates.append(end_time)
                        results_replicates.append(result_actives_found)
                    
                    #results of all replicates are appended
                    influence_learning_results.append(results_replicates)
                    influence_learning_times.append(times_replicates)
                    print(f"[eval]: Processing dataset: {name} via {influence_calc}, {regression_calc}, {sampling_calc} took {end_time} seconds")
                    print("#####################################")
                    print("#####################################")
        
        #the performance of the state of the art greedy approach and the alternative,
        #random sampling, are calculated
        greedy_results = []
        greedy_times = []
        random_results = []
        random_times = []
        
        for rep in range(replicates):
            #again, actives are counted and then converted to percentage of
            #all actives in the dataset
            print("[eval]: Calculating greedy and random performance in replicate ", rep)
            start_greedy = time.time()
            greedy_actives = greedy_actives_found(x_train, y_p_train, steps = steps, step_size = step_size, random_size = random_size, seed = seeds[rep])
            greedy_actives = greedy_actives/y_p_train.sum()
            greedy_results.append(greedy_actives)
            greedy_time = time.time() - start_greedy
            greedy_times.append(greedy_time)
            
            start_random = time.time()
            random_actives = random_actives_found(x_train, y_p_train, steps = steps, step_size = step_size, random_size = random_size, seed = seeds[rep])
            random_actives = random_actives/y_p_train.sum()
            random_results.append(random_actives)
            random_time = time.time() - start_random
            random_times.append(random_time)
            
            
        #the results are plotted and saved to a csv file
        plot_learning(combined_names, influence_learning_results, greedy_results, random_results, y_p_train, filename, name, influence, steps = steps, step_size = step_size)            
        plot_learning_bands(combined_names, influence_learning_results, greedy_results, random_results, y_p_train, filename, name, influence, steps = steps, step_size = step_size)            
        save_data(combined_names, influence_learning_results, greedy_results, random_results,y_p_train, influence_learning_times, greedy_times, random_times,filename, name, influence, steps = steps, step_size = step_size, replicates=replicates)
  
    
    
if __name__ == "__main__":
    main(dataset = args.dataset,
         representation = args.representation.lower(),
         influence = args.influence,
         regression = args.regression,
         sampling = args.sampling,
         replicates = args.replicates,
         steps = args.steps,
         step_size = args.step_size,
         random_size = args.random_size,
         filename = args.filename)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
