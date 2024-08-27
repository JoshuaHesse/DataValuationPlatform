#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:53:03 2023

@author: joshua
"""
from typing import *
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import pandas as pd
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from typing import List
from typing import List
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np
from joblib import Parallel, delayed
import os

from ActiveLearning.utils_jh import *
import numpy as np
from rdkit import Chem
import pandas as pd
import argparse
import os
import time
import pickle

parser = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default="all",
                    help="Which dataset from ../Datasets to use for the analysis, options: [all, specific_name]")

parser.add_argument('--representation', default="ECFP",
                    help="Which sample representation to use for the run, options: [ECFP, RDKIT]")


args = parser.parse_args()

#-----------------------------------------------------------------------------#

"""
        This pipeline can be used to calculate rdkit or ecfp representations 
        of datasets and export the resulting arrays to .pkl files for future
        use.
        Unless specified otherwhise, the arrays will be saved in the Datasets_descr 
        folder in a folder named after the respective dataset


"""

def main(
        dataset,
        representation,
        ):
    
    #either all datasets are processed, or a specific one
    if dataset != "all":
        dataset_names = [dataset]
    else:
        dataset_names = os.listdir("../Datasets")    
        dataset_names = [x for x in dataset_names]
        
    
    for i in range(len(dataset_names)):

        #load i-th dataset training and validation set separately and get mols
        print("----------------------")
        name = dataset_names[i]
        print(f"[Descriptor_calc]: Processing dataset: {name}")
        db_train = pd.read_csv("../Datasets/" + name + "/" + name + "_train.csv")
        db_val = pd.read_csv("../Datasets/" + name + "/" + name + "_val.csv")
        
        mols_train = list(db_train["SMILES"])
        mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
        
        mols_val = list(db_val["SMILES"])
        mols_val = [Chem.MolFromSmiles(x) for x in mols_val]  
        
        
        #calculate respective representation
        if(representation == "rdkit"):
            
            #when calculating rdkit descriptors, the resulting values are scaled
            #via a StandardScaler
            scaler = StandardScaler()
            print(f"[Descriptor_calc]: Calculating RDKIT desriptors")
            x_train = get_rdkit_descr(mols_train)
            x_val = get_rdkit_descr(mols_val)
    
            #when using rdkit molec descriptors, the vectors have to be normalized after creation
            scaler.fit(x_train)
            print(f"[Descriptor_calc]: Scaling the descriptors")
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)

        else:
            print(f"[Descriptor_calc]: Calculating ECFP desriptors")
            x_train = get_ECFP(mols_train)
            x_val = get_ECFP(mols_val)
        
        #if this dataset has no saved descriptors yet, a new file is created
        if not(os.path.isdir("../Datasets_descr/" + name)):
            os.mkdir("../Datasets_descr/" + name)
        #save training set dataframe
        path = "../Datasets_descr/" + name + "/" + name + "_" + representation + "_train.pkl"
        pd.DataFrame(x_train).to_pickle(path)
        print(f"[Descriptor_calc]: File saved at {path}")

        #save validation set dataframe
        path = "../Datasets_descr/" + name + "/" + name + "_" + representation + "_val.pkl"
        pd.DataFrame(x_val).to_pickle(path)
        print(f"[Descriptor_calc]: File saved at {path}")
        
        
if __name__ == "__main__":
    main(
        dataset = args.dataset,
        representation = args.representation.lower()
        )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
