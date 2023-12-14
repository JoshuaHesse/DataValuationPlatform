#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:53:03 2023

@author: joshua
"""
import pandas as pd
import deepchem as dc
import pandas as pd
import argparse
import os
import time
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from typing import List
import numpy as np
import rdkit

parser = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--group_type', default="disease",
                    help="Which dataset group type, options: [disease, target]")


parser.add_argument('--dataset_group', default="fungal",
                    help="Which dataset group from Moldata to use for the descr export, options: [None, specific_name]")


parser.add_argument('--representation', default="ECFP",
                    help="Which sample representation to use for the run, options: [ECFP, SMILES, RDKIT]")


args = parser.parse_args()

#-----------------------------------------------------------------------------#

"""
        This pipeline can be used to calculate rdkit or ecfp representations 
        of datasets and export the resulting arrays to .pkl files for future
        use.
        Unless specified otherwhise, the arrays will be saved in the Datasets_descr 
        folder in a folder named after the respective dataset


"""

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
    array = np.empty((len(mols), len(descriptor_names)), dtype=np.float32)
    
    # Create a molecular descriptor calculator
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    # Loop through each molecule and compute descriptors
    for i, mol in enumerate(mols):
        # Calculate descriptors for the current molecule
        temp = calc.CalcDescriptors(mol)
        array[i] = temp
        
    # Clip extremely large or small values to a predefined range and substitute nan for 0
    np.nan_to_num(array, copy=False, nan=0.0, posinf=10e8, neginf=-10e8)

    return array

def main(
        group_type,
        dataset_group,
        representation,
        ):
    
    #either all datasets are processed, or a specific one
    
    print("Extraction started")
    
    if group_type == "disease":
        mapping = pd.read_csv("MolData/Data/aid_disease_mapping.csv")
    else:
        mapping = pd.read_csv("MolData/Data/aid_target_mapping.csv")
    
    db_moldata = pd.read_csv("MolData/Data/all_molecular_data.csv")
    
    mapping = mapping[["AID", dataset_group]]
        
    mapping = mapping.loc[mapping[dataset_group] == 1]

    aids = mapping["AID"].tolist()
    
    aids.insert(0, "smiles")
    
    db_moldata = db_moldata[aids]
    
    db_moldata_no_nan = db_moldata.dropna(axis=0, thresh=2)
    
    db_moldata_no_nan.reset_index(drop=True, inplace=True)
    if(representation == "ecfp"):
        featurizer = dc.feat.CircularFingerprint(size=1024, chiral = True)
    
        features = featurizer.featurize(db_moldata_no_nan["smiles"])
    elif(representation=="rdkit"):

        #when calculating rdkit descriptors, the resulting values are scaled
        #via a StandardScaler
        scaler = StandardScaler()
        print(f"[Descriptor_calc]: Calculating RDKIT desriptors")
        mols_train = list(db_moldata_no_nan["smiles"])
        mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
        x_train = get_rdkit_descr(mols_train)


        #when using rdkit molec descriptors, the vectors have to be normalized after creation
        scaler.fit(x_train)
        print(f"[Descriptor_calc]: Scaling the descriptors")
        features = scaler.transform(x_train)


    aids = aids[1:]

    
    
    
    print("[eval]: Beginning Description Calculation...")
    print("[eval]: Run parameters:")
    print(f"        group_type: {group_type}")
    print(f"        dataset_group: {dataset_group}")
    print(f"        dataset_ids: {aids}")
    print(f"        representation: {representation}")

    
    if not(os.path.isdir("../Datasets_descr_MolData/" + dataset_group)):
           os.mkdir("../Datasets_descr_MolData/" + dataset_group)
    
    for i in aids:
        df = db_moldata_no_nan.dropna(subset=[i])
        features_tmp = features[df.index]
        name = i
        path = "../Datasets_descr_MolData/" + dataset_group +"/" + name + "_" + representation + ".pkl"
        pd.DataFrame(features_tmp).to_pickle(path)


        
if __name__ == "__main__":
    main(
        group_type = args.group_type,
        dataset_group = args.dataset_group,
        representation = args.representation.lower()
        )
        
        
        
        
        
        
        
        
        
        
        
        
        
        