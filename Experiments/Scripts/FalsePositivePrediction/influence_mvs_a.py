#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:21:04 2023

@author: joshua

The function in this file is used to predict false positives using Minimal 
Variance Sampling Analysis (MVS-A)
"""
import pandas as pd
import numpy as np
from utils_jh import process_ranking, bedroc_score, enrichment_factor_score, get_scaffold_rate, run_logger
from sklearn.metrics import precision_score
import rdkit
from MVS_A import sample_analysis
from typing import Tuple, List
import time

def run_mvsa(
        mols: List[rdkit.Chem.rdchem.Mol],
        x: np.ndarray,
        y_p: np.ndarray,
        y_f: np.ndarray,
        y_c: np.ndarray,
        idx: List[int],
        replicates: int,
        log_predictions: bool = True
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes MVS-A analysis on given dataset
    
    Uses MVS-A importance scores to rank primary screen actives in terms of 
    likelihood of being false positives or true positives. Top ranked compounds
    are FPs, as indicated in TracIn, while bottom ranked compounds are TPs.
    Finally, the function computes precision@90, EF10, BEDROC20
    and scaffold diversity metrics.

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
    
    #create results containers
    temp = np.zeros((replicates,9))
    logs = pd.DataFrame([])
    
    #loop analysis over replicates
    for j in range(replicates):
        
        #create MVS-A object (ONLY FEEDS PRIMARY DATA, CONFIRMATORY NEVER
        #OBSERVED BY THE MODEL DURING THIS ANALYSIS)
        obj = sample_analysis(x = x,
                              y = y_p, 
                              params = "default",
                              verbose = False,
                              seed = j)
        
        #get sample importances and measure time
        start = time.time()
        obj.get_model()
        vals = obj.get_importance()
        temp[j,0] = time.time() - start
        
        #convert importances into labels (top 90% -> FP, bottom 10% -> TP)
        flags, flags_alt = process_ranking(y_p, vals)

        #get precision@90 for FP and TP retrieval
        temp[j,1] = precision_score(y_f, flags[idx])
        temp[j,2] = precision_score(y_c, flags_alt[idx])
        
        #get EF10 for FPs and TPs
        temp[j,3] = enrichment_factor_score(y_f, flags[idx])
        temp[j,4] = enrichment_factor_score(y_c, flags_alt[idx])

        #get BEDROC20 for FPs and TPs
        temp[j,5] = bedroc_score(y_f, vals[idx], reverse=True)
        temp[j,6] = bedroc_score(y_c, vals[idx], reverse=False)	 
        
        #get scaffold diversity for compounds that got flagged as FPs
        idx_fp = np.where(flags == 1)[0]
        mols_fp = [mols[x] for x in idx_fp]
        temp[j,7] = get_scaffold_rate(mols_fp)

        #get scaffold diversity for compounds that got flagged as TPs
        idx_tp = np.where(flags_alt == 1)[0]
        mols_tp = [mols[x] for x in idx_tp]
        temp[j,8] = get_scaffold_rate(mols_tp)
    
        #optionally fill up logger
        if log_predictions is True:
            logs_temp = run_logger(mols, idx, y_f, y_c, flags[idx],
                                    flags_alt[idx], vals[idx], "mvsa")
            logs = pd.concat([logs, logs_temp], axis=0, ignore_index=True)
          
    return temp, logs