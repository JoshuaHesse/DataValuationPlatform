
import numpy as np
import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from typing import List, Tuple
from rdkit import Chem, DataStructs
import os

# ##############################################################################

def run_logger(
        mols: List[rdkit.Chem.rdchem.Mol],
        idx: List[int],
        y_f: np.ndarray,
        y_c: np.ndarray,
        flags: np.ndarray,
        flags_alt: np.ndarray,
        score: np.ndarray,
        algorithm: str
        ) -> pd.DataFrame:
    """Logs raw predictions for a given algorithm

    Args:
        mols:       (M,) mol objects from primary data
        idx:        (V,) positions of primary actives with confirmatory
                    read´out
        y_f:        (V,) false positive labels (1=FP)
        y_c:        (V,) true positive labels (1=FP)
        flags:      (V,) FP predictions
        flags_alt:  (V,) TP predictions
        algorithm:  name of the algorithm for FP/TP detection
    
    Returns:
        Dataframe (V,5) containing SMILES, true labels and raw predictions
    """
    
    #get primary actives with confirmatory measurement
    mols_subset = [mols[x] for x in idx]

    #store results in db
    smiles = [Chem.MolToSmiles(x) for x in mols_subset]
    db = pd.DataFrame({
        "SMILES": smiles,
        "False positives": y_f,
        "True positives": y_c,
        "FP - " + algorithm: flags,
        "TP - " + algorithm: flags_alt,
        "Score - " + algorithm: score
        })
    
    s = pd.DataFrame({"SMILES": ["next replicate", "starting_now"],
    "False positives": [0.0,0.0],
    "True positives": [0.0,0.0],
    "FP - " + algorithm: [0.0,0.0],
    "TP - " + algorithm: [0.0,0.0],
    "Score - " + algorithm: [0.0,0.0]
    })
    
    db = pd.concat([db, s], axis=0, ignore_index=True)
    return db


def get_ECFP(
        mols: List[rdkit.Chem.rdchem.Mol],
        radius:int = 2,
        nbits:int = 1024
        ) -> np.ndarray:
    """Calculates ECFPs for given set of molecules
    
    Args:
        mols:   (M,) mols to compute ECFPs  
        radius: radius for fragment calculation
        nbits:  bits available for folding ECFP

    Returns:
        array (M,1024) of ECFPs 
    """
    #create empty array as container for ecfp
    array = np.empty((len(mols), nbits), dtype=np.float32)

    #get ecfps in list
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius, nbits) for x in mols]
    
    #store each element in list into array via cython
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])
    return array

# -----------------------------------------------------------------------------#

def get_rdkit_descr(mols: List[rdkit.Chem.rdchem.Mol]
                    ) -> np.ndarray:
    
    """Calculates RDKIT mol. descriptors for a given set of molecules
           Args:
              mols:   (M,) mols to compute RDKIT descriptors  

             Returns:
              array (M, 208) of RDKIT descriptors 
    """
    
    # Get descriptor names
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    
    # Initialize an array to store descriptor values
    results = np.empty((len(mols), len(descriptor_names)), dtype=np.float32)
    
    # Create a molecular descriptor calculator
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    # Loop through each molecule and compute descriptors
    for i, mol in enumerate(mols):
        # Calculate descriptors for the current molecule
        temp = calc.CalcDescriptors(mol)
        results[i] = temp
        
    # Clip extremely large or small values to a predefined range and substitute nan for 0
    np.nan_to_num(results, copy=False, nan=0.0, posinf=10e8, neginf=-10e8)

    return results



#-----------------------------------------------------------------------------#    
def get_smiles_list(mols: List[rdkit.Chem.rdchem.Mol]
            ) -> List[str]:    
    
    """    
    Calculates SMILES for a given set of molecules.
    Args:
       mols:   (M,) mols to compute RDKIT descriptors  

   Returns:
       list (M,) of SMILES strings 
    
    """

    return [Chem.MolToSmiles(mol) for mol in mols]



#-----------------------------------------------------------------------------#
def get_scaffold_rate(
        mols: List[rdkit.Chem.rdchem.Mol]
        ) -> float:
    """Computes scaffold diversity for given set of molecules
    
    Args:
        mols:   (M,) mols to check for scaffold diversity

    Returns:
        percentage of unique Murcko scaffolds in the set, as the number of
        unique Murcko scaffolds divided by the number of molecules
    """
    
    #safety check (can happen with datasets with very low % of
    #primary actives in confirmatory dataset)
    if len(mols) > 0:

        #count mols, create empty Murcko scaffold list
        tot_mols = len(mols)
        scaffs = [0]*len(mols)

        #check SMILES for Murcko scaffold
        smiles = [Chem.MolToSmiles(x) for x in mols]
        for i in range(len(mols)):
            scaffs[i] = MurckoScaffold.MurckoScaffoldSmiles(smiles[i])
        
        #remove scaffold duplicates and compare to mol count
        scaffs = list(set(scaffs))
        n_scaffs = len(scaffs)
        rate = n_scaffs * 100 / tot_mols
    else:
        rate = 0.0

    return rate

# -----------------------------------------------------------------------------#

def get_labels(
        dataframe: pd.DataFrame,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """Fetches all necessary labels for AIC analysis
    
    Args:
        dataframe:  (M,3) table with merged primary and confirmatory assay 
                    information

    Returns:
        A tuple containing:
            1. (M,) Primary screen labels
            2. (V,) FP labels for primary actives with confirmatory readout
            3. (V,) TP labels for primary actives with confirmatory readout
            4. (V,) Position of primary actives with confirmatory readout inside
            primary screen data

    """

    #primary labels for training ML models
    y_p = np.array(dataframe["Primary"])

    #get slice with compounds with confirmatory measurement
    selected_rows = dataframe[~dataframe['Confirmatory'].isnull()]
    
    #further cut slice to select only compounds that were primary actives
    selected_rows = selected_rows.loc[selected_rows['Primary'] == 1]
    
    #confirmatory readout becomes TP vector (primary matches confirmatory)
    y_c = np.array(selected_rows["Confirmatory"])

    #FP vector as opposite of TP vector
    y_f = (y_c - 1) * -1

    #position of final compounds in primary screen data
    idx = np.array(selected_rows.index)
    
    return y_p, y_c, y_f, idx

# -----------------------------------------------------------------------------#

def get_labels_val(
        dataframe: pd.DataFrame,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """Fetches all necessary labels for AIC analysis
    
    Args:
        dataframe:  (M,3) table with merged primary and confirmatory assay 
                    information

    Returns:
        A tuple containing:
            1. (M,) Primary screen labels
            2. (V,) FP labels for primary actives with confirmatory readout
            3. (V,) TP labels for primary actives with confirmatory readout
            4. (V,) Position of primary actives with confirmatory readout inside
            primary screen data

    """

    #primary labels for training ML models
    y_p = np.array(dataframe["Primary"])

    #get slice with compounds with confirmatory measurement
    selected_rows = dataframe[~dataframe['Confirmatory'].isnull()]
    
    #further cut slice to select only compounds that were primary actives
    
    
    #confirmatory readout becomes TP vector (primary matches confirmatory)
    y_c = np.array(selected_rows["Confirmatory"])

    #FP vector as opposite of TP vector
    y_f = (y_c - 1) * -1

    #position of final compounds in primary screen data
    idx = np.array(selected_rows.index)
    
    return y_p, y_c, y_f, idx

#-----------------------------------------------------------------------------#



def process_ranking(
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
    scores_pos = vals_box[idx_pos]
    
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


# ----------------------------------------------------------------------------#

def enrichment_factor_score(
        y_true: np.ndarray,
        y_pred: np.ndarray
        ) -> float:
    """
    Function to compute Enrichment Factor using precomputed binary labels
    according to the threshold set in process_ranking
    """
    
    compounds_at_k = np.sum(y_pred)
    if compounds_at_k==0:
        return 0
    total_compounds = len(y_true)
    total_actives = np.sum(y_true)
    tp_at_k = len(np.where(y_true + y_pred == 2)[0])

    return (tp_at_k / compounds_at_k) * (total_actives / total_compounds)

# -----------------------------------------------------------------------------#

def bedroc_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reverse: bool = True
        ) -> float:
    """
    Function to compute BEDROC score from raw rankings, using alpha=20 as
    default. Adapted from https://github.com/deepchem
    """
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    scores = list(zip(y_true, y_pred))
    scores = sorted(scores, key=lambda pair: pair[1], reverse=reverse)
    
    output = CalcBEDROC(scores = scores,
                        col = 0,
                        alpha = 20)

    return output

#-----------------------------------------------------------------------------#


def store_row_without_replicates(
        analysis_array: np.ndarray,
        dataset_array: np.ndarray,
        fp_rate: float,
        tp_rate: float,
        index: int
        ) -> np.ndarray:
    """Stores i-th dataset results in performance container for X datasets
    
    Args:
        analysis_array: (X,20) dataframe that stores results of a given
                        algorithm for all datasets
        dataset_array:  (1,9) array with the results of a given algorithm on
                        the i-th dataset
        fp_rate:        fraction of false positives in the confirmatory dataset
        tp_rate:        fraction of true positives in the confirmatory dataset
        index:          i-th row position to store results in

    Returns:
        Updated analysis array with results stored in the correct row
    """

    analysis_array[index, 0] = np.mean(dataset_array[:,0])      #mean training time
    analysis_array[index, 1] = np.std(dataset_array[:,0])       #STD training time

    analysis_array[index, 2] = fp_rate                          #baseline FP rate
    analysis_array[index, 3] = np.mean(dataset_array[:,1])      #mean precision@90 FP
    analysis_array[index, 4] = np.std(dataset_array[:,1])       #STD precision@90 FP
    
    analysis_array[index, 5] = tp_rate                          #baseline TP rate
    analysis_array[index, 6] = np.mean(dataset_array[:,2])      #mean precision@90 TP
    analysis_array[index, 7] = np.std(dataset_array[:,2])       #STD precision@90 TP

    analysis_array[index, 8] = np.mean(dataset_array[:,3])      #EF10 for FP
    analysis_array[index, 9] = np.std(dataset_array[:,3])       #STD EF10 FP

    analysis_array[index, 10] = np.mean(dataset_array[:,4])     #EF10 for TP
    analysis_array[index, 11] = np.std(dataset_array[:,4])      #STD EF10 TP

    analysis_array[index, 12] = np.mean(dataset_array[:,5])     #BEDROC20 for FP
    analysis_array[index, 13] = np.std(dataset_array[:,5])      #STD

    analysis_array[index, 14] = np.mean(dataset_array[:,6])     #BEDROC20 for TP
    analysis_array[index, 15] = np.std(dataset_array[:,6])      #STD

    analysis_array[index, 16] = np.mean(dataset_array[:,7])     #means FP scaffold diversity
    analysis_array[index, 17] = np.std(dataset_array[:,7])      #STD FP scaffold diversity

    analysis_array[index, 18] = np.mean(dataset_array[:,8])     #means TP scaffold diversity
    analysis_array[index, 19] = np.std(dataset_array[:,8])      #STD TP scaffold diversity
    
    return analysis_array


#------------------------------------------------------------------------------#

def store_row(
        analysis_array: np.ndarray,
        dataset_array: np.ndarray,
        replicates: int,
        fp_rate: float,
        tp_rate: float,
        index: int
        ) -> np.ndarray:
    """Stores i-th dataset results in performance container for X datasets
    
    Args:
        analysis_array: (X,20) dataframe that stores results of a given
                        algorithm for all datasets
        dataset_array:  (1,9) array with the results of a given algorithm on
                        the i-th dataset
        replicates:     number of replicates to be saved
        fp_rate:        fraction of false positives in the confirmatory dataset
        tp_rate:        fraction of true positives in the confirmatory dataset
        index:          i-th row position to store results in

    Returns:
        Updated analysis array with results stored in the correct row (not the
        most elegant solution but at least it provides a straightforward way
        to handle both single and multi dataset performance collection)
    """
    pos = 0
    analysis_array[index, pos] = np.mean(dataset_array[:,0])      #mean training time
    pos += 1
    analysis_array[index, pos] = np.std(dataset_array[:,0])       #STD training time
    pos +=1
    end_pos = pos + replicates
    analysis_array[index, pos:end_pos] = dataset_array[:,0]
    pos = end_pos
    
    analysis_array[index, pos] = fp_rate 
    pos +=1                         #baseline FP rate
    analysis_array[index, pos] = np.mean(dataset_array[:,1])      #mean precision@90 FP
    pos +=1
    analysis_array[index, pos] = np.std(dataset_array[:,1])       #STD precision@90 FP
    pos +=1
    end_pos = pos + replicates
    analysis_array[index, pos:end_pos] = dataset_array[:,1]
    pos = end_pos
    
    analysis_array[index, pos] = tp_rate                          #baseline TP rate
    pos +=1
    analysis_array[index, pos] = np.mean(dataset_array[:,2])      #mean precision@90 TP
    pos +=1
    analysis_array[index, pos] = np.std(dataset_array[:,2])       #STD precision@90 TP
    pos +=1
    end_pos = pos + replicates
    analysis_array[index, pos:end_pos] = dataset_array[:,2]
    pos = end_pos
    
    analysis_array[index, pos] = np.mean(dataset_array[:,3])      #EF10 for FP
    pos +=1
    analysis_array[index, pos] = np.std(dataset_array[:,3])       #STD EF10 FP
    pos +=1
    end_pos = pos + replicates
    analysis_array[index, pos:end_pos] = dataset_array[:,3]
    pos = end_pos
    
    analysis_array[index, pos] = np.mean(dataset_array[:,4])     #EF10 for TP
    pos +=1
    analysis_array[index, pos] = np.std(dataset_array[:,4])      #STD EF10 TP
    pos +=1
    end_pos = pos + replicates
    analysis_array[index, pos:end_pos] = dataset_array[:,4]
    pos = end_pos
    
    analysis_array[index, pos] = np.mean(dataset_array[:,5])     #BEDROC20 for FP
    pos +=1
    analysis_array[index, pos] = np.std(dataset_array[:,5])      #STD
    pos +=1
    end_pos = pos + replicates
    analysis_array[index, pos:end_pos] = dataset_array[:,5]
    pos = end_pos
    
    analysis_array[index, pos] = np.mean(dataset_array[:,6])     #BEDROC20 for TP
    pos +=1
    analysis_array[index, pos] = np.std(dataset_array[:,6])      #STD
    pos +=1
    end_pos = pos + replicates
    analysis_array[index, pos:end_pos] = dataset_array[:,6]
    pos = end_pos
    
    analysis_array[index, pos] = np.mean(dataset_array[:,7])     #means FP scaffold diversity
    pos +=1
    analysis_array[index, pos] = np.std(dataset_array[:,7])      #STD FP scaffold diversity
    pos +=1
    end_pos = pos + replicates
    analysis_array[index, pos:end_pos] = dataset_array[:,7]
    pos = end_pos
    
    analysis_array[index, pos] = np.mean(dataset_array[:,8])     #means TP scaffold diversity
    pos +=1
    analysis_array[index, pos] = np.std(dataset_array[:,8])      #STD TP scaffold diversity
    pos +=1
    analysis_array[index, pos:] = dataset_array[:,8]
    return analysis_array
#-----------------------------------------------------------------------------#

def save_results_without_replicates(
        results: List[np.ndarray],
        dataset_names: List,
        representation: str,
        filename: str,
        filter_type: str
        ) -> None:
    """Saves results from all algorithms to their respective .csv files. Additionally,
    saves a summary.csv file that encompasses the means for FP and TP prediction
    according to precision@90, EF10 and BEDROC20 of all algorithms to allow easy 
    inter-algorithm comparison.
    all .csv files are saved in a new folder named filename within the Results folder
    
    Args:
        results:        list (5,) containing results arrays for all algorithms.
                        In case one algorithm was not selected for the run, it
                        is stored as an empty array in this list and it will 
                        not be saved to .csv
        dataset_names:  list (X,) of all dataset names analysed in the run
        filename:       common name of the .csv files to use when saving (i.e.
                        if filename=output, the .csv with MVS-A results will be
                        saved as "mvsa_output.csv")
        filter_type:    structural alerts name to append when saving the performance
                        of fragment filters (i.e. "filter_PAINS_output.csv")

    Returns:
        None
    """
    
    column_names = [
                "Time - mean", "Time - STD",
                "FP rate",
                "FP Precision@90 - mean", "FP Precision@90 - STD",
                "TP rate",
                "TP Precision@90 - mean", "TP Precision@90 - STD",
                "FP EF10 - mean", "FP EF10 - STD",
                "TP EF10 - mean", "TP EF10 - STD",
                "FP BEDROC20 - mean", "FP BEDROC20 - STD",
                "TP BEDROC20 - mean", "TP BEDROC20 - STD",
                "FP Scaffold - mean", "FP Scaffold - STD",
                "TP Scaffold - mean", "TP Scaffold - STD"
                ]
    
    #creates new folder in the Results folder for this set of results
    if not(os.path.isdir("../Results/" + filename)):
        os.mkdir("../Results/" + filename)
    prefix = "../Results/" + filename + "/"
    suffix = ".csv"
    algorithm = ["score_", "filter_" + filter_type + "_"]
    
    #creates a summary Dataframe that is used to concatenate all summary-data
    summary = pd.DataFrame(index = dataset_names)
    #temporary DataFrames to splice out from each full result DataFrame and add to summary
    fp_prec = pd.DataFrame(index = dataset_names)
    tp_prec = pd.DataFrame(index = dataset_names)
    fp_ef = pd.DataFrame(index = dataset_names)
    tp_ef = pd.DataFrame(index = dataset_names)
    fp_bed = pd.DataFrame(index = dataset_names)
    tp_bed = pd.DataFrame(index = dataset_names) 
    
    for i in range(len(results)):
        if np.sum(results[i]) != 0:             #save only if array is not empty
            db = pd.DataFrame(
                    data = results[i],
                    index = dataset_names,
                    columns = column_names
                    )

            

            #Columns of interest are spliced out of each result DataFrame, renamed to 
            #corresponding algorithm and added to a temporary DataFrame of the respective
            #performance measurement summary
            fp_p = db.loc[:,["FP Precision@90 - mean"]]
            fp_p.rename(columns={"FP Precision@90 - mean": "FP Precision@90 - mean_" + algorithm[i]}, inplace = True)           
            fp_prec = pd.concat([fp_prec, fp_p], axis=1)
            
            tp_p = db.loc[:,["TP Precision@90 - mean"]]
            tp_p.rename(columns={"TP Precision@90 - mean": "TP Precision@90 - mean_" + algorithm[i]}, inplace = True)           
            tp_prec = pd.concat([tp_prec, tp_p], axis=1)
            
            fp_e = db.loc[:,["FP EF10 - mean"]]
            fp_e.rename(columns={"FP EF10 - mean": "FP EF10 - mean_" + algorithm[i]}, inplace = True)           
            fp_ef = pd.concat([fp_ef, fp_e], axis=1)
            
            tp_e = db.loc[:,["TP EF10 - mean"]]
            tp_e.rename(columns={"TP EF10 - mean": "TP EF10 - mean_" + algorithm[i]}, inplace = True)           
            tp_ef = pd.concat([tp_ef, tp_e], axis=1)
            
            fp_b = db.loc[:,["FP BEDROC20 - mean"]]
            fp_b.rename(columns={"FP BEDROC20 - mean": "FP BEDROC20 - mean_" + algorithm[i]}, inplace = True)           
            fp_bed = pd.concat([fp_bed, fp_b], axis=1)
            
            tp_b = db.loc[:,["TP BEDROC20 - mean"]]
            tp_b.rename(columns={"TP BEDROC20 - mean": "TP BEDROC20 - mean_" + algorithm[i]}, inplace = True)           
            tp_bed = pd.concat([tp_bed, tp_b], axis=1)

            #full results are saved in the Results/filename folder for each algorithm individually
            db.to_csv(prefix + algorithm[i] + filename + "_" + representation + suffix)
            
    #all temporary summary-DataFrames are combined to one final one, that is then
    #saved in the Results/filename folder
    summary = pd.concat([summary, fp_prec, tp_prec, fp_ef, tp_ef, fp_bed, tp_bed], axis=1)
    summary.to_csv(prefix + filename + "_" + "summary_score_filter" + suffix)


def save_results(
        results: List[np.ndarray],
        dataset_names: List,
        representation: str,
        replicates: int,
        filename: str,
        filter_type: str
        ) -> None:
    """Saves results from all algorithms to their respective .csv files. Additionally,
    saves a summary.csv file that encompasses the means for FP and TP prediction
    according to precision@90, EF10 and BEDROC20 of all algorithms to allow easy 
    inter-algorithm comparison.
    all .csv files are saved in a new folder named filename within the Results folder
    
    Args:
        results:        list (5,) containing results arrays for all algorithms.
                        In case one algorithm was not selected for the run, it
                        is stored as an empty array in this list and it will 
                        not be saved to .csv
        dataset_names:  list (X,) of all dataset names analysed in the run
        replicates:     number of replicates recorded
        filename:       common name of the .csv files to use when saving (i.e.
                        if filename=output, the .csv with MVS-A results will be
                        saved as "mvsa_output.csv")
        filter_type:    structural alerts name to append when saving the performance
                        of fragment filters (i.e. "filter_PAINS_output.csv")

    Returns:
        None
    """
    
        
    # Add column names for replicates
    column_names = [
        "Time - mean", "Time - STD"]
    for j in range(replicates): # Each of the five replicates
        column_names.append("Time - Replicate {0}".format(j+1))
    column_names.append("FP rate")
    column_names.append("FP Precision@90 - mean")
    column_names.append("FP Precision@90 - STD")
    for j in range(replicates): # Each of the five replicates
        column_names.append("FP Precision@90 - Replicate {0}".format(j+1))
    column_names.append("TP rate")
    column_names.append("TP Precision@90 - mean")
    column_names.append("TP Precision@90 - STD")
    for j in range(replicates): # Each of the five replicates
        column_names.append("TP Precision@90 - Replicate {0}".format(j+1))
    column_names.append("FP EF10 - mean")
    column_names.append("FP EF10 - STD")
    for j in range(replicates): # Each of the five replicates
        column_names.append("FP EF10 - Replicate {0}".format(j+1))
    column_names.append("TP EF10 - mean")
    column_names.append("TP EF10 - STD")
    for j in range(replicates): # Each of the five replicates
        column_names.append("TP EF10 - Replicate {0}".format(j+1))
    column_names.append("FP BEDROC20 - mean")
    column_names.append("FP BEDROC20 - STD")
    for j in range(replicates): # Each of the five replicates
        column_names.append("FP BEDROC20 - Replicate {0}".format(j+1))
    column_names.append("TP BEDROC20 - mean")
    column_names.append("TP BEDROC20 - STD")
    for j in range(replicates): # Each of the five replicates
        column_names.append("TP BEDROC20 - Replicate {0}".format(j+1))
    column_names.append("FP Scaffold- mean")
    column_names.append("FP Scaffold - STD")
    for j in range(replicates): # Each of the five replicates
        column_names.append("FP Scaffold - Replicate {0}".format(j+1))
    column_names.append("TP Scaffold- mean")
    column_names.append("TP Scaffold - STD")
    for j in range(replicates): # Each of the five replicates
        column_names.append("TP Scaffold - Replicate {0}".format(j+1))
    
    #creates new folder in the Results folder for this set of results
    if not(os.path.isdir("../Results/" + filename)):
           os.mkdir("../Results/" + filename)

    prefix = "../Results/" + filename + "/"
    suffix = ".csv"
    algorithm = ["mvsa_", "catboost_self","catboost_test", "knn_", "dvrl_", "tracin_", "tracin_pos_", "tracin_self_"]
    
    #creates a summary Dataframe that is used to concatenate all summary-data
    summary = pd.DataFrame(index = dataset_names)
    #temporary DataFrames to splice out from each full result DataFrame and add to summary
    fp_prec = pd.DataFrame(index = dataset_names)
    tp_prec = pd.DataFrame(index = dataset_names)
    fp_ef = pd.DataFrame(index = dataset_names)
    tp_ef = pd.DataFrame(index = dataset_names)
    fp_bed = pd.DataFrame(index = dataset_names)
    tp_bed = pd.DataFrame(index = dataset_names) 
    
    for i in range(len(results)):
        if np.sum(results[i]) != 0:             #save only if array is not empty
            db = pd.DataFrame(
                    data = results[i],
                    index = dataset_names,
                    columns = column_names
                    )
# =============================================================================
# this is the code to just summarize everying together - without reorganizing the summary DataFrame
#             db_cut = db.loc[:,["FP Precision@90 - mean", "TP Precision@90 - mean", "FP EF10 - mean", "TP EF10 - mean",
#                          "FP BEDROC20 - mean", "TP BEDROC20 - mean"]]
#             db_cut.rename(columns = {"FP Precision@90 - mean": "FP Precision@90 - mean_" + algorithm[i], "TP Precision@90 - mean":"TP Precision@90 - mean_" + algorithm[i],
#                                      "FP EF10 - mean": "FP EF10 - mean_" + algorithm[i],
#                                      "TP EF10 - mean":"TP EF10 - mean_" + algorithm[i],
#                                      "FP BEDROC20 - mean": "FP BEDROC20 - mean_" + algorithm[i], 
#                                      "TP BEDROC20 - mean":"TP BEDROC20 - mean_" + algorithm[i]},
#                                       inplace = True)
# =============================================================================
            

            #Columns of interest are spliced out of each result DataFrame, renamed to 
            #corresponding algorithm and added to a temporary DataFrame of the respective
            #performance measurement summary
            fp_p = db.loc[:,["FP Precision@90 - mean"]]
            fp_p.rename(columns={"FP Precision@90 - mean": "FP Precision@90 - mean_" + algorithm[i]}, inplace = True)           
            fp_prec = pd.concat([fp_prec, fp_p], axis=1)
            
            tp_p = db.loc[:,["TP Precision@90 - mean"]]
            tp_p.rename(columns={"TP Precision@90 - mean": "TP Precision@90 - mean_" + algorithm[i]}, inplace = True)           
            tp_prec = pd.concat([tp_prec, tp_p], axis=1)
            
            fp_e = db.loc[:,["FP EF10 - mean"]]
            fp_e.rename(columns={"FP EF10 - mean": "FP EF10 - mean_" + algorithm[i]}, inplace = True)           
            fp_ef = pd.concat([fp_ef, fp_e], axis=1)
            
            tp_e = db.loc[:,["TP EF10 - mean"]]
            tp_e.rename(columns={"TP EF10 - mean": "TP EF10 - mean_" + algorithm[i]}, inplace = True)           
            tp_ef = pd.concat([tp_ef, tp_e], axis=1)
            
            fp_b = db.loc[:,["FP BEDROC20 - mean"]]
            fp_b.rename(columns={"FP BEDROC20 - mean": "FP BEDROC20 - mean_" + algorithm[i]}, inplace = True)           
            fp_bed = pd.concat([fp_bed, fp_b], axis=1)
            
            tp_b = db.loc[:,["TP BEDROC20 - mean"]]
            tp_b.rename(columns={"TP BEDROC20 - mean": "TP BEDROC20 - mean_" + algorithm[i]}, inplace = True)           
            tp_bed = pd.concat([tp_bed, tp_b], axis=1)


# =============================================================================
#           summary = pd.concat([summary, db_cut], axis=1)
# =============================================================================
            #full results are saved in the Results/filename folder for each algorithm individually
            db.to_csv(prefix + algorithm[i] + filename + "_" + representation + suffix)
            
    #all temporary summary-DataFrames are combined to one final one, that is then
    #saved in the Results/filename folder
    summary = pd.concat([summary, fp_prec, tp_prec, fp_ef, tp_ef, fp_bed, tp_bed], axis=1)
    summary.to_csv(prefix + filename + "_" + "summary_algorithms" + suffix)

   
