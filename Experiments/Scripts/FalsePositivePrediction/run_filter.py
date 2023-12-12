#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:07:37 2023

@author: joshua

This file contains the functions to predict false positives based on 
fragment filters.

"""
import pandas as pd
import numpy as np
from utils_jh import bedroc_score, enrichment_factor_score, get_scaffold_rate, run_logger
from sklearn.metrics import precision_score
import rdkit
import time
from typing import Tuple, List
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def run_filter(
        mols: List[rdkit.Chem.rdchem.Mol],
        idx: List[int],
        filter_type: str,
        y_f: np.ndarray,
        y_c: np.ndarray,
        log_predictions: bool = True
        ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Executes structural alert analysis on given dataset
    
    Uses structural alerts to mark primary hits as TPs or FPs. Then, 
    it computes precision@90, EF10, BEDROC20 and scaffold diversity indices. 
    
    Args:
        mols:               (M,) mol objects from primary data
        idx:                (V,) positions of primary actives with confirmatory    
                            readout  
        filter_type:        name of the structural alerts class to use                   
        y_f:                (V,) false positive labels (1=FP)        
        y_c:                (V,) true positive labels (1=FP)
        log_predictions:    enables raw predictions logging
     
    Returns:
        Tuple containing one array (1,9) with precision@90 for FP and TP retrieval, 
        scaffold diversity and training time, and one dataframe (V,5) with
        SMILES, true labels and raw predictions
    """

    #create results containers
    temp = np.zeros((1,9))
    logs = pd.DataFrame([])   

    #get primary actives with confirmatory measurement
    mol_subset = [mols[x] for x in idx]
            
    #use substructure filters to flag FPs, time it and measure precision
    start = time.time()
    flags = filter_predict(mol_subset, filter_type)
    temp[0,0] = time.time() - start
    temp[0,1] = precision_score(y_f, flags)
    
    #invert filter flagging to find TPs and measure precision
    flags_alt = (flags - 1) * -1
    temp[0,2] = precision_score(y_c, flags_alt)
    
    #get EF10 for FPs and TPs
    temp[0,3] = enrichment_factor_score(y_f, flags)
    temp[0,4] = enrichment_factor_score(y_c, flags_alt)

    #get BEDROC20 for FPs and TPs
    temp[0,5] = bedroc_score(y_f, flags, reverse=True)
    temp[0,6] = bedroc_score(y_c, flags_alt, reverse=True)	

    #get scaffold diversity for compounds that got flagged as FPs
    idx_fp = np.where(flags == 1)[0]
    mols_fp = [mol_subset[x] for x in idx_fp]
    temp[0,7] = get_scaffold_rate(mols_fp)
    
    #get scaffold diversity for compounds that got flagged as TPs
    idx_tp = np.where(flags_alt == 1)[0]
    mols_tp = [mol_subset[x] for x in idx_tp]
    temp[0,8] = get_scaffold_rate(mols_tp)

    #optionally fill up logger
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags,
                                flags_alt, flags, "filter")
            
    return temp, logs  





def filter_predict(
        mols: List[rdkit.Chem.rdchem.Mol],
        catalog_name: str
        ) -> List[int]:
    """Uses structural alerts to predict whether compounds are TP or FP
    
    Args:
        mols:           (M,) molecules to predict
        catalog_name:   name of the structural alerts set to use for
                        predictions

    Returns:
        list (M,) of predictions according to chosen structural alert
    """
    #create RDKIT filter catalog dictionary (could be expanded)
    catalogs = {
        "PAINS":    FilterCatalogParams.FilterCatalogs.PAINS,
        "PAINS_A":  FilterCatalogParams.FilterCatalogs.PAINS_A,
        "PAINS_B":  FilterCatalogParams.FilterCatalogs.PAINS_B,
        "PAINS_C":  FilterCatalogParams.FilterCatalogs.PAINS_C,
        "NIH":      FilterCatalogParams.FilterCatalogs.NIH
        }
    
    #create substructure checker according to fragment set
    params = FilterCatalogParams()
    params.AddCatalog(catalogs[catalog_name])
    catalog = FilterCatalog(params)

    #check all mols and flag ones that have a match
    verdict = np.zeros((len(mols),))
    for i in range(len(mols)):
        if mols[i] is not None:
            entry = catalog.GetFirstMatch(mols[i])
            if entry is not None:
                verdict[i] = 1
    
    return verdict
    