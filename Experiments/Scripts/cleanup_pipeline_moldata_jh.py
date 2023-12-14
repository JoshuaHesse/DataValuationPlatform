"""Dataset preprocessing script.
Adjustment of the cleanup_pipeline to extract, clean, and export 1 AID dataset from
the MolData Dataset. 
"""
import os
import pandas as pd
from rdkit import Chem
import numpy as np
from chembl_structure_pipeline import standardizer
import sys
import argparse
import math

###############################################################################

parser = argparse.ArgumentParser(description=__doc__,
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--AID',
                    help="AID number of the assay")		

parser.add_argument('--filename', default=None,
                    help="Name to use when saving the processed dataset")


args = parser.parse_args()

###############################################################################

#Preprocessing function to handle duplicate measurements in one assay
def f_1(x):
    if x > 0.5:
        return 1
    else:
        return 0
    
###############################################################################

def main(
        AID,
        filename
        ):
    
    #print info of run
    print("[cleanup]: Beginning dataset cleanup run...")
    print(f"[cleanup]: Currently merging AID {AID}")
    if filename == None:
        filename = AID
    print(f"[cleanup]: Dataset will be saved as {filename}.csv")
   
    #create preprocessing log container
    logs = []

    #load and remove NaN SMILES
    t1 = pd.read_csv("MolData/Data/all_molecular_data.csv", low_memory=False)

    #get smiles
    smiles = list(t1["smiles"])
    split = list(t1["split"])
    #cut off possible non-string last rows
    for i in range(len(smiles)):
        if isinstance(smiles[i], str) is True:
            lim_1 = i
            break
    smiles = smiles[lim_1:]
    
    #get labels
    column_name = "activity_" + AID
    act_1 = list(t1[column_name])[lim_1:]
    
    

    #convert to mol, sanitize and store original dataset sizes
    mols_1 = [Chem.MolFromSmiles(x, sanitize=True) for x in smiles]
   
    n_original_1 = len(mols_1)
    n_act_1 = sum(act_1)


    #filter NoneType
    idx = [x for x in list(range(len(mols_1))) if mols_1[x] is not None]
    mols_1 = [mols_1[x] for x in idx]
    act_1 = [act_1[x] for x in idx]
    split = [split[x] for x in idx]
   
   

    #remove salts
    mols_1 = [standardizer.get_parent_mol(x)[0] for x in mols_1]
  
    #return to SMILES
    smiles_1 = [Chem.MolToSmiles(x) for x in mols_1]
   

    

    #remove duplicates by aggregating according to consensus
    db_1 = pd.DataFrame({"SMILES": smiles_1, "Activity":act_1, "Split":split})   
    db_1 = db_1.groupby(["SMILES"], as_index=False).mean()
    
    #merge and remove molecules that somehow were constructed incorrectly
    db_1["Mols"] = [Chem.MolFromSmiles(x) for x in list(db_1["SMILES"])]
    db_1.dropna(inplace=True, subset=["Mols"])
    db_1.drop(["Mols"], inplace=True, axis=1)
    #merge and remove molecules that somehow were constructed incorrectly
    
    #count train_test_val_split
    n1 = db_1["split"].value_counts().index[0]
    n2 = db_1["split"].value_counts().index[1]
    n3 = db_1["split"].value_counts().index[2]
    x1 = db_1["Split"].value_counts()[0]
    x2 = db_1["Split"].value_counts()[1]
    x3 = db_1["Split"].value_counts()[2]
    
    
    
    
    
    #log differences before/after preprocessing
    n_samples = len(db_1)
    logs.append("Dataset generated using AID " + AID)
    logs.append("Compounds before preprocessing: " + str(n_original_1))
    logs.append("Actives before preprocessing: " + str(n_act_1))
    logs.append("Compounds after preprocessing: " + str(n_samples))
    logs.append("Actives after preprocessing: " + str(np.sum(db_1["Activity"])))
   
    logs.append("The following split was taken from the original moldata dataset: " )
    logs.append(n1 + ": " + str(x1))
    logs.append(n2 + ": " + str(x2))
    logs.append(n3 + ": " + str(x3))

    

        
    #test full dataset
    #current tests check that all confirmatory compounds were present before in the primary,
    #and that all duplicate records in either assay have been merged / removed
    
    
    #save training set dataframe
 
    path = "../MolData_cleaned_Datasets/" + filename + ".csv"
    db_1.to_csv(path)
    print(f"[cleanup]: File saved at {path}")

  
    #save logs
    logpath = "../Logs_MolData/cleanup/" + filename + ".txt"
    with open(logpath, 'w') as f:
        for line in logs:
            f.write(f"{line}\n")
    print(f"[cleanup]: Log saved at {logpath}")


if __name__ == "__main__":
    main(
        AID = args.AID,
        filename = args.filename
        )






