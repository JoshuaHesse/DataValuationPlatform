#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:58:31 2023

@author: joshua


This pipeline can be used to compare undersampling using influence functions
with random undersampling. The pipeline is designed to be applied to the 
Moldata datasets, testing one group of datasets at a time. The methods that can
be compared are MVSA, TracIn (self-influence), Catboost (self-influence), and 
random undersampling.

The pipeline trains the influence model with the training set, removes the highest
highest or lowest scoring samples (either in- or excluding actives) and tests
performance on the validation set. This is done for a range of 5% up to 95% of
the training set samples being removed.
"""
import argparse
import tensorflow as tf
from influence_functions import influence_functions_dict
from model_comparison import run_model_comparison
import os
import numpy as np
import pandas as pd



parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_group', default="fungal",
                    help="Which dataset from ../Datasets to use for the analysis, options: [all, specific_name]")

parser.add_argument('--influence', default="MVSA",
                    help="Which Influence function to use, options: [MVSA, TracIn_self, Catboost_self, Random]")

parser.add_argument('--inactives_only', default=True, type=bool,
                    help="Whether to delete only inactives or all types of samples, options: [True, False]")

parser.add_argument('--gpu', default=0, type=int,
                    help="Which GPU to use for the analysis, options: [0,1]")

parser.add_argument('--replicates', default=5, type=int,
                    help="Number of replicates per dataset")

parser.add_argument('--filename', default="output",
                    help="Name to use when saving performance results")

args = parser.parse_args()



def main(dataset_group,
         influence,
         filename, 
         replicates,
         gpu,
         inactives_only):
    
    if influence.lower() == "tracin_self":
        #setting the GPU to the GPU set in the arguments
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_visible_devices(physical_devices[gpu], "GPU")
        tf.config.experimental.set_memory_growth(physical_devices[gpu], True)
        
    #collecting the datasets on which to run the pipeline  
    datasets = os.listdir("../../../Datasets_descr_MolData/"+dataset_group)
    #removing the suffix to only get the name of the assay ("activity_123456")
    dataset_names = []
    for i in datasets:
        if(i[-9:] == "_ecfp.pkl"):
            dataset_names.append(i[:-9])

   
    for rep in range(replicates):
        
        results_total = []
        counter = 0
        #run domain adaptation for each dataset and collect the results during it
        for name in dataset_names:
            counter +=1
            print("----------------------")
            print("Dataset number " + str(counter) + " of " + str(len(dataset_names)) + " in replicate " + str(rep))
            print("Analysis of Dataset " + name + " of dataset_group " + dataset_group+ " in replicate " + str(rep))
            db_moldata = pd.read_csv("MolData/Data/all_molecular_data.csv", low_memory = False)
            #import precalc features
            features_path = "../../Datasets_descr_MolData/"+dataset_group+"/" + name + "_ecfp.pkl"
            features = pd.read_pickle(features_path)
            features = features.to_numpy()
        
            #slice assay out of moldata dataset
            assay_name = name
            assay = db_moldata[["smiles", assay_name, "split"]]
            assay = assay.dropna(axis=0)
            assay.reset_index(drop=True, inplace=True)
            train_indices = assay[assay['split'] == 'train'].index.tolist()
            test_indices = assay[assay['split'] == 'test'].index.tolist()
            validation_indices = assay[assay['split'] == 'validation'].index.tolist()
            
                
            print("Train indices:", len(train_indices))
            print("Test indices:", len(test_indices))
            print("Validation indices:", len(validation_indices))
            
            #separate train, test, and val labels
            train_labels = assay[assay['split'] == 'train'][assay_name].values
            test_labels = assay[assay['split'] == 'test'][assay_name].values
            val_labels = assay[assay['split'] == 'validation'][assay_name].values
            
        
            #separate train, test, and val features
            train_features = features[train_indices]
            test_features = features[test_indices]
            val_features = features[validation_indices]
            
           
        
            print("Train - test - val : ", str(len(train_labels)), ",", str(len(test_labels)), ",", str(len(val_labels)))
            
            if influence != "random":
                train_influences  = influence_functions_dict[influence.lower()](x_train = train_features,y_train = train_labels, i = i, seed = rep)
            
            else:
                train_influences = None
            results_list = []
            
            #set count of removed samples to range of 5% up to 95 % of the 
            #training set size
            remove_count = []
    
            remove_count = []        
            for rm in range(19):
                rm = rm+1
                remove_count.append(int(len(train_labels)*0.05 * rm))
            
            #run domain adaptation on each of the remove_counts and save the data
            #together with the number of removed samples
            for i in remove_count:
                if (i > len(train_labels)):
                    break
                print("Running model comparison with ", i, " samples removed in replicate " + str(rep))
                results = run_model_comparison(x_train=train_features, y_train=train_labels, x_test=test_features,
                                               y_test=test_labels, x_val=val_features, y_val=val_labels,
                                               influences=train_influences, influence_method=influence, num_removed=i,replicate=rep, inactives_only=inactives_only, class_weights=True)
        
                results_list.append((i, results))
            results_total.append(results_list)
        
            
        #export the percentages as indices for the data saving
        percentages = []
        for rm in range(19):
            rm = rm+1
            percentages.append(round(0.05 * rm, 3))
            
        #translate the results into a list of dataframes
        dataframe_list = []
        for i in results_total:
            index = percentages
            counts = [x[0] for x in i]
            basic = [x[1][0] for x in i]
            high = [x[1][1] for x in i]
            low = [x[1][2] for x in i]
            df = pd.DataFrame(index = index, data = {"counts": counts, "basic": basic, "high": high, "low": low})
            dataframe_list.append(df)
          
        #export the results all into 1 .csv file, containing the performances of
        #all models, as well as the number of 
        name = "../../Results/Undersampling/"+filename +"_" + str(rep) + ".csv"
        with open(name, 'w') as file:
            for i, df in enumerate(dataframe_list):
                file.write(dataset_names[i]+'\n')
                df.to_csv(file, mode='a', header=True)
                file.write('\n')
        print("===================================")
        print("saved at " + name)
    
if __name__ == "__main__":
    main(dataset_group = args.dataset_group,
         influence = args.influence.lower(),
         filename = args.filename,
         gpu = args.gpu,
         replicates = args.replicates,
         inactives_only = args.inactives_only) 
    
    
    
    
