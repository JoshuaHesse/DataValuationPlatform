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
from utils_jh import bedroc_score, enrichment_factor_score, get_scaffold_rate, run_logger, process_ranking
from sklearn.metrics import precision_score
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdmolops
import time
from typing import Tuple, List
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def process_ranking_filters(
        y: np.ndarray,
        vals_box: List[float],
        percentile: int = 90
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Converts raw importance scores in binary predictions using percentiles
    
    Args:
        y:          (M,) primary screen labels
        vals_box:   (M,) raw importance scores
        percentile: value to use for thresholding

    Returns:
        A tuple containing two arrays (M,) with labels indicating whether
        compounds are TPs or FPs according to the importances predicted by
        the ML model. First element of the tuple are the indexes of compounds
        who have a score >90%, the second element are the ones <10%.
    """

    #select importance scores from primary actives
    idx_pos = np.where(y == 1)[0]
    scores_pos = vals_box
    
    #compute top 90% and bottom 10% percentile thresholds. since this is done
    #only by looking primary screening data, it is completely unbiased against
    #leakage from confirmatory screen data
    top10 = np.percentile(scores_pos, percentile)
    bottom10 = np.percentile(scores_pos, 100 - percentile)
    
    #create empty arrays (proponent = compound above 90% threshold)
    proponent = np.zeros((len(y),))
    opponent = np.zeros((len(y),))
    
    #find primary actives which fall outside of either threshold
    idx_pro = np.where(scores_pos >= top10)[0]
    idx_opp = np.where(scores_pos <= bottom10)[0]
    
    #fill respective arrays with respective labels. in this context,
    #proponents are samples >90%, while opponents are <10%
    proponent[idx_pos[idx_pro]] = 1
    opponent[idx_pos[idx_opp]] = 1
    
    return proponent, opponent
def run_filter(
        mols: List[rdkit.Chem.rdchem.Mol],
        idx: List[int],
        filter_type: str,
        y_p: np.ndarray,
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
    if filter_type in ["PAINS", "PAINS_A", "PAINS_B", "PAINS_C", "NIH"]:
        flags_short = pains_predict(mol_subset, filter_type)
        flags = np.zeros((len(y_p),))
        expanded_vals = np.zeros((len(y_p),))
        for i, x in enumerate(idx):
            flags[x] = flags_short[i]
            expanded_vals[x] = flags_short[i]
        flags_alt = (flags - 1) * -1
    else:
        vals = filter_predict(mol_subset)
        expanded_vals = np.zeros((len(y_p),))
        for i, x in enumerate(idx):
            expanded_vals[x] = vals[i]
        #flags = np.where(flags > 0, 1, 0)   
        flags, flags_alt = process_ranking(y_p, expanded_vals)
    temp[0,0] = time.time() - start
    temp[0,1] = precision_score(y_f, flags[idx])
    
    #invert filter flagging to find TPs and measure precision
    #flags_alt = (flags - 1) * -1
    temp[0,2] = precision_score(y_c, flags_alt[idx])
    
    #get EF10 for FPs and TPs
    temp[0,3] = enrichment_factor_score(y_f, flags[idx])
    temp[0,4] = enrichment_factor_score(y_c, flags_alt[idx])

    #get BEDROC20 for FPs and TPs
    temp[0,5] = bedroc_score(y_f, expanded_vals[idx], reverse=True)
    temp[0,6] = bedroc_score(y_c, expanded_vals[idx], reverse=True)	

    #get scaffold diversity for compounds that got flagged as FPs
    idx_fp = np.where(flags == 1)[0]
    mols_fp = [mols[x] for x in idx_fp]
    temp[0,7] = get_scaffold_rate(mols_fp)
    
    #get scaffold diversity for compounds that got flagged as TPs
    idx_tp = np.where(flags_alt == 1)[0]
    mols_tp = [mols[x] for x in idx_tp]
    temp[0,8] = get_scaffold_rate(mols_tp)

    #optionally fill up logger
    if log_predictions is True:
        logs = run_logger(mols, idx, y_f, y_c, flags[idx],
                                flags_alt[idx], flags[idx], "filter")
            
    return temp, logs  





def pains_predict(
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


def filter_predict(
        mols: List[rdkit.Chem.rdchem.Mol]
        ) -> List[int]:
    """Uses structural alerts (REOS and GSK) to predict whether compounds 
    are TP or FP. Sources:
        https://www.sciencedirect.com/science/article/pii/S2472555222068800#ec1
        https://www.nature.com/articles/nrd1063#Sec3    
    
    Args:
        mols:           (M,) molecules to predict

    Returns:
        list (M,) of predictions according to the structural alert. The integer
        denotes the number of alerts triggered by the compound.
    """
    
    filter_db = pd.read_csv("filters.csv")
    smarts = list(filter_db["SMARTS"])
    checks = [Chem.MolFromSmarts(x) for x in smarts]
    [Chem.GetSSSR(x) for x in checks]
    
    structural_alerts = np.zeros(len(mols))
    for i in range(len(mols)):
        for j in range(len(checks)):
            if mols[i].HasSubstructMatch(checks[j]):
                structural_alerts[i] += 1
    
    physchem_alerts = np.zeros(len(mols))
    for i in range(len(mols)):
        mw = Descriptors.ExactMolWt(mols[i])
        logp = Crippen.MolLogP(mols[i])
        hbd = rdMolDescriptors.CalcNumHBD(mols[i])
        hba = rdMolDescriptors.CalcNumHBA(mols[i])
        charge = rdmolops.GetFormalCharge(mols[i])
        rb = rdMolDescriptors.CalcNumRotatableBonds(mols[i])
        ha = mols[i].GetNumHeavyAtoms()
        if mw < 200 or mw > 500:
            physchem_alerts[i] += 1
        if logp < -5.0 or logp > 5.0:
            physchem_alerts[i] += 1
        if hbd > 5:
            physchem_alerts[i] += 1
        if hba > 10:
            physchem_alerts[i] += 1
        if charge < -2 or charge > 2:
            physchem_alerts[i] += 1
        if rb > 8:
            physchem_alerts[i] += 1
        if ha < 15 or ha > 50:
            physchem_alerts[i] += 1
    
    return structural_alerts + physchem_alerts   
    
