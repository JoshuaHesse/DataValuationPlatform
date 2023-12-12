#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:58:28 2023

@author: joshua

this file contains the function to use the score method to predict false positives
"""
import pandas as pd
import numpy as np
from utils_jh import process_ranking, bedroc_score, enrichment_factor_score, get_scaffold_rate, run_logger
from sklearn.metrics import precision_score
import rdkit
from typing import Tuple, List



def run_score(
    df: pd.DataFrame,
    mols: List[rdkit.Chem.rdchem.Mol],
    idx: np.ndarray,
    y_p: np.ndarray,
    y_f: np.ndarray,
    y_c: np.ndarray,
    log_predictions: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes Score analysis on given dataset

    Uses raw primary screen scores (as indicated in PUBCHEM_ACTIVITY_SCORE) to 
    rank primary screen actives in terms of likelihood of being false positives 
    or true positives. Top ranked compounds (most active primaries) are TPs, 
    bottom ranked compounds FPs. Finally, the function computes precision@90, 
    EF10, BEDROC20 and scaffold diversity metrics.
    
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
    temp = np.zeros((1,9))
    logs = pd.DataFrame([])
    
    #get scores
    scores = np.array(df["Score"])
            
    #convert into labels (top 90% -> TP, bottom 10% -> FP)
    flags, flags_alt = process_ranking(y_p, scores)
        
    #get precision@90 for FPs and TPs
    temp[0,1] = precision_score(y_f, flags_alt[idx])
    temp[0,2] = precision_score(y_c, flags[idx])
    
    #get EF10 for FPs and TPs
    temp[0,3] = enrichment_factor_score(y_f, flags_alt[idx])
    temp[0,4] = enrichment_factor_score(y_c, flags[idx])

    #get BEDROC20 for FPs and TPs
    temp[0,5] = bedroc_score(y_f, scores[idx], reverse=False)
    temp[0,6] = bedroc_score(y_c, scores[idx], reverse=True)
    
    #get diversity for FPs
    idx_fp = np.where(flags_alt == 1)[0]
    mols_fp = [mols[x] for x in idx_fp]
    temp[0,7] = get_scaffold_rate(mols_fp)
        
    #get diversity for TPs
    idx_tp = np.where(flags == 1)[0]
    mols_tp = [mols[x] for x in idx_tp]
    temp[0,8] = get_scaffold_rate(mols_tp)

    #optionally fill up logger
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags_alt[idx],
                                    flags[idx], scores[idx], "score")
                
    return temp, logs  
