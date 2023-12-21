#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:30:08 2023

@author: joshuahesse
"""
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
import logging
import argparse
from pathlib import Path
import sys
from typing import List
import os
import numpy as np
from chembl_structure_pipeline import standardizer
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from DataValuationPlatform.models.utils.utils_jh import *
from shutil import rmtree
logging.basicConfig(stream=sys.stdout, level=20)


def get_project_root() -> Path:
    return Path(__file__).parent.parent



class HTSDataPreprocessor:
    """
        The `HTSDataPreprocessor` class is designed to manage and process chemical datasets, specifically those involving molecular data. It is equipped with various methods to load raw data, preprocess it, and create molecular descriptors that are crucial for computational chemistry and cheminformatics tasks. Below is an overview of its functionality:

        Attributes:
        - dataset_names: A list of names for the datasets being handled.
        - datasets: A dictionary mapping dataset names to their respective data containers.

        Methods:
        - load_data(dataset_name): Loads raw data files for the specified dataset.
        - clean_primary_data(data_frame): Cleans the primary data from the raw dataset.
        - clean_confirmatory_data(data_frame): Cleans the confirmatory data from the raw dataset.
        - combine_datasets(dataset_name): Combines primary and confirmatory datasets into one.
        - split_datasets(dataset_name): Splits the dataset into training and validation sets.
        - preprocess_data(): Applies the preprocessing steps to all datasets in `dataset_names`.
        - save_data(filename): Saves the processed training and validation data to CSV files.
        - create_descriptors(descriptor_type): Creates molecular descriptors such as SMILES, ECFP, or RDKit descriptors.
        - load_preprocessed_data(): Loads preprocessed training and validation datasets from CSV files.
        - load_precalculated_descriptors(representation): Loads precalculated molecular descriptors from pickled files.
        - get_dataset(dataset_name): Retrieves the data container for a specified dataset.

        Each method and attribute is part of a workflow to ensure that chemical datasets are ready for further analysis or machine learning tasks. Descriptors are mathematical representations of molecules that are necessary for predictive modeling in cheminformatics.
        """
    def __init__(
            self, 
            dataset_names: List[str],
            validation_split: float = 0.1):
        """
        Initializes the HTSDataPreprocessor class.
        
        Args:
            dataset_names: A list of names for the datasets to be processed.
            validation_split: A float representing the proportion of the dataset to be used as the validation set.
        
        Returns:
            None.
        """
        if type(dataset_names) != list:
            logging.error("Please enter the dataset names as a list, even for single datasets. Example: [GPCR_3]")
        self.dataset_names = dataset_names
        self.datasets = {}
        path_to_root = get_project_root()
        log_path = path_to_root / "Logs" / "dataset_metadata.csv"
        self.metadata = pd.read_csv(log_path)[["Codename", "Primary HTS AID", "Confirmatory HTS AID"]]
        self.validation_split = validation_split

    class DatasetContainer:
        def __init__(
                self, 
                dataset_name: str):
            """
              initializes the DatasetContainer object 
    
                Parameters
                ----------
                dataset_name : str
                    DESCRIPTION.
    
                Returns
                -------
                None.
    
            """
            self.dataset_name = dataset_name
            self.data_1 = {}
            self.data_2 = {}
            self.combined_data = {}
            self.training_set = {}
            self.validation_set = {}
            self.training_set_descriptors = {}
            self.validation_set_descriptors = {}
            self.training_set_labels = {}
            self.validation_set_labels = {}
            self.representation = None

    def create_custom_dataset(
            self,
            dataset_name: str,
            training_set_smiles: List[str],
            training_set_labels,
            training_set_confirmatory_labels=None,
            validation_set_smiles: List[str] = None,
            validation_set_labels = None,
            validation_split: float = None,
            training_set_descriptors = None,
            validation_set_descriptors = None):
        """
        Creates and adds a custom dataset to the HTSDataPreprocessor. This function allows users to input their own data, 
         including SMILES strings, corresponding labels, and optionally confirmatory labels for the training set. 
         If a validation set is not provided, the function splits the training set into training and validation sets 
         while maintaining a similar ratio of active to inactive samples.
        
         Parameters:
         - dataset_name (str): Name of the new dataset.
         - training_set_smiles (List[str]): List of SMILES strings for the training set.
         - training_set_labels: Labels corresponding to the training set SMILES strings.
         - training_set_confirmatory_labels (optional): Confirmatory labels for the training set. Defaults to None.
         - validation_set_smiles (List[str], optional): List of SMILES strings for the validation set. Defaults to None.
         - validation_set_labels (optional): Labels corresponding to the validation set SMILES strings. Defaults to None.
         - validation_split (float, optional): Proportion of the training set to be used as validation set if no validation set is provided. Defaults to None.
         - training_set_descriptors (optional): Descriptors corresponding to the training set SMILES strings. Defaults to None.
         - validation_set_descriptors (optional): Descriptors corresponding to the validation set SMILES strings. Defaults to None.
        
         Returns:
         None. The function updates the internal state of the HTSDataPreprocessor instance by adding the new custom dataset.
         
    
         Note: If confirmatory labels are supplied for the training set and no validation set is provided, 
         the validation set is created using only samples with confirmatory labels. The size of the validation 
         set is determined by the validation_split ratio applied to the number of samples with confirmatory labels.
        """
        # Create or get existing dataset container and add dataset name to the list
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = self.DatasetContainer(dataset_name)
            self.dataset_names.append(dataset_name)
    
        # Initialize training dataset with optional confirmatory labels
        training_data = {'SMILES': training_set_smiles, 'Primary': training_set_labels}
        if training_set_confirmatory_labels is not None:
            training_data['Confirmatory'] = training_set_confirmatory_labels
        self.datasets[dataset_name].training_set = pd.DataFrame(training_data)
    
        # Set training set descriptors if provided
        if training_set_descriptors is not None:
            self.datasets[dataset_name].training_set_descriptors = training_set_descriptors
    
        # Handle validation set
        if validation_set_smiles is not None and validation_set_labels is not None:
            # User provided custom validation set
            self.datasets[dataset_name].validation_set = pd.DataFrame({'SMILES': validation_set_smiles, 'Confirmatory': validation_set_labels})
            if validation_set_descriptors is not None:
                self.datasets[dataset_name].validation_set_descriptors = validation_set_descriptors
        else:
            # Create validation set from samples with confirmatory labels if provided
            if validation_split is None:
                validation_split = self.validation_split
    
            if training_set_confirmatory_labels is not None:
                # Filter dataset to include only samples with confirmatory labels
                confirmatory_data = self.datasets[dataset_name].training_set.dropna(subset=['Confirmatory'])
                val_size = int(len(confirmatory_data) * validation_split)
                train_indices, val_indices = train_test_split(
                    confirmatory_data.index, 
                    test_size=val_size, 
                    stratify=confirmatory_data['Confirmatory'])
    
                self.datasets[dataset_name].validation_set = self.datasets[dataset_name].training_set.loc[val_indices].reset_index(drop=True)
                self.datasets[dataset_name].training_set = self.datasets[dataset_name].training_set.loc[train_indices].reset_index(drop=True)
            else:
                # Standard split when no confirmatory labels are provided
                train_indices, val_indices = train_test_split(
                    range(len(self.datasets[dataset_name].training_set)), 
                    test_size=validation_split, 
                    stratify=training_set_labels)
    
                self.datasets[dataset_name].validation_set = self.datasets[dataset_name].training_set.iloc[val_indices].reset_index(drop=True)
                self.datasets[dataset_name].training_set = self.datasets[dataset_name].training_set.iloc[train_indices].reset_index(drop=True)
                self.datasets[dataset_name].validation_set.rename(columns={'Primary': 'Confirmatory'}, inplace=True)

             # Resetting training and validation set labels
            self.datasets[dataset_name].training_set_labels = self.datasets[dataset_name].training_set['Primary']
            self.datasets[dataset_name].validation_set_labels = self.datasets[dataset_name].validation_set['Confirmatory']
           
            # Splitting descriptors if provided
            if training_set_descriptors is not None:
                self.datasets[dataset_name].validation_set_descriptors = training_set_descriptors[val_indices]
                self.datasets[dataset_name].training_set_descriptors = training_set_descriptors[train_indices]

        
    def load_data(
            self, 
            dataset_name: str,
            path_to_raw: str,
            ):
        """
        Loads the primary and confirmatory data for a given dataset from the metadata file.
        If you want to add your own datasets, add them to the metadata file and download the raw data to the Raw_data folder
        
        Args:
            dataset_name: The name of the dataset to load.
        
        Returns:
            None. Updates the datasets attribute with the loaded data.
        """
        try:
            if dataset_name not in self.datasets:
                self.datasets[dataset_name] = self.DatasetContainer(dataset_name)
            aids = self.metadata[self.metadata['Codename'] == dataset_name][["Primary HTS AID", "Confirmatory HTS AID"]].values[0]
            if len(aids) != 2:
                raise ValueError("Expected exactly 2 AIDs for the dataset.")
            filename_1 = path_to_raw + f"AID_{str(aids[0])}_datatable_all.csv"
            filename_2 = path_to_raw + f"/AID_{str(aids[1])}_datatable_all.csv"
            self.datasets[dataset_name].data_1 = pd.read_csv(filename_1, low_memory=False)
            self.datasets[dataset_name].data_2 = pd.read_csv(filename_2, low_memory=False)
            logging.info(f"Datasets loaded successfully for {dataset_name}")
        except Exception as e:
            logging.error(f"An error occurred while loading the datasets for {dataset_name}: {e}")
            sys.exit(1)

    def add_dataset_by_AID(
            self, 
            codename: str,
            primary_AID: str,
            confirmatory_AID: str):
        """
        Adds a new dataset to the HTSDataPreprocessor by specifying two Assay IDs (AIDs) and a codename. 
        This method appends these details to the metadata file and updates the internal dataset collection.
    
        Parameters:
        - codename (str): The codename for the new dataset.
        - primary_AID (str): The AID for the primary assay.
        - confirmatory_AID (str): The AID for the confirmatory assay.
    
        The method updates the metadata file with the new dataset details and then loads the data based on the provided AIDs.
    
        Returns:
        None. The metadata file is updated, and the new dataset is loaded into the HTSDataPreprocessor instance.
        """
        # Append new dataset information to the metadata file
        new_metadata = pd.DataFrame({
        "Codename": [codename],
        "Primary HTS AID": [primary_AID],
        "Confirmatory HTS AID": [confirmatory_AID]})
        current_metadata = self.metadata
        self.metadata = pd.concat([current_metadata, new_metadata], ignore_index=True)
        if codename not in self.datasets:
                self.datasets[codename] = self.DatasetContainer(codename)
                self.dataset_names.append(codename)
        
        


    @staticmethod
    def process_duplicates(x: float):
        """
        Processes Activity Labels if compounds are present multiple times in a dataset

        Parameters
        ----------
        x : float
            The mean activity label for compounds (usually 1 or 0, but can be different if the dataset contains duplicates).

        Returns
        -------
        int
            Activity label averaged over all instances of the compund.

        """
        if x > 0.5:
            return 1
        else:
            return 0


    def clean_primary_data(
            self, 
            t1: pd.DataFrame):
        """
        Cleans and processes the primary dataset.
        
        Args:
            t1: The primary dataset as a pandas DataFrame.
        
        Returns:
            A cleaned and processed version of the primary dataset as a pandas DataFrame.
        """
        if t1 is not None:
            t1.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"], inplace=True)
            smiles_1 = list(t1["PUBCHEM_EXT_DATASOURCE_SMILES"])
            for i in range(len(smiles_1)):
                if isinstance(smiles_1[i], str) is True:
                    lim_1 = i
                    break
            smiles_1 = smiles_1[lim_1:]
            act_1 = list(t1["PUBCHEM_ACTIVITY_OUTCOME"])[lim_1:]
            score_1 = list(t1["PUBCHEM_ACTIVITY_SCORE"])[lim_1:]
            
            #turn smiles into mols
            mols_1 = [Chem.MolFromSmiles(x, sanitize=True) for x in smiles_1]
            idx = [x for x in list(range(len(mols_1))) if mols_1[x] is not None]
            mols_1 = [mols_1[x] for x in idx]
            act_1 = [act_1[x] for x in idx]
            score_1 = [score_1[x] for x in idx]
            #standardize mols
            mols_1 = [standardizer.get_parent_mol(x)[0] for x in mols_1]
            #turn mols back into smiles for standardized smiles representation
            smiles_1 = [Chem.MolToSmiles(x) for x in mols_1]
            idx = [x for x in list(range(len(act_1))) if act_1[x] == "Active"]
            act_1 = np.zeros((len(act_1),))
            act_1[idx] = 1
            #eliminate duplicates
            db_1 = pd.DataFrame({"SMILES": smiles_1, "Primary":act_1, "Score":score_1})
            db_1 = db_1.groupby(["SMILES"], as_index=False).mean()
            db_1["Primary"] = db_1["Primary"].apply(HTSDataPreprocessor.process_duplicates)
            print("Primary data cleaning done successfully.")
            return db_1
        else:
            print("Data not loaded. Please load data before cleaning.")
            return None

    def clean_confirmatory_data(
            self, 
            t2: pd.DataFrame):
        """
        Cleans and processes the confirmatory dataset.
        
        Args:
            t2: The confirmatory dataset as a pandas DataFrame.
        
        Returns:
            A cleaned and processed version of the confirmatory dataset as a pandas DataFrame.
        """
        if t2 is not None:
            t2.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"], inplace=True)
            smiles_2 = list(t2["PUBCHEM_EXT_DATASOURCE_SMILES"])
            for i in range(len(smiles_2)):
                if isinstance(smiles_2[i], str) is True:
                    lim_2 = i
                    break
            smiles_2 = smiles_2[lim_2:]
            act_2 = list(t2["PUBCHEM_ACTIVITY_OUTCOME"])[lim_2:]
            #turn smiles into mols
            mols_2 = [Chem.MolFromSmiles(x, sanitize=True) for x in smiles_2]
            n_original_2 = len(mols_2)
            n_act_2 = act_2.count("Active")
            idx = [x for x in list(range(len(mols_2))) if mols_2[x] is not None]
            mols_2 = [mols_2[x] for x in idx]
            act_2 = [act_2[x] for x in idx]
            #standardize mols
            mols_2 = [standardizer.get_parent_mol(x)[0] for x in mols_2]
            #turn mols back into smiles for standardized smiles representation
            smiles_2 = [Chem.MolToSmiles(x) for x in mols_2]
            idx = [x for x in list(range(len(act_2))) if act_2[x] == "Active"]
            act_2 = np.zeros((len(act_2),))
            act_2[idx] = 1
            #eliminate duplicates
            db_2 = pd.DataFrame({"SMILES": smiles_2, "Confirmatory":act_2})
            db_2 = db_2.groupby(["SMILES"], as_index=False).mean()
            db_2["Confirmatory"] = db_2["Confirmatory"].apply(HTSDataPreprocessor.process_duplicates)
            print("Confirmatory data cleaning done successfully.")
            return db_2
        else:
            print("Data not loaded. Please load data before cleaning.")
            return None


    def combine_datasets(
            self,
            dataset_name: str):
        """
        Combines the primary and confirmatory datasets into one.
        
        Args:
            dataset_name (str): The name of the dataset to be combined.
        
        Returns:
            None. Updates the combined_data attribute in the dataset container.
        """
        if self.datasets[dataset_name].data_1 is not None and self.datasets[dataset_name].data_2 is not None:
            # combines the primary and confirmatory datasets into one dataframe
            self.datasets[dataset_name].combined_data = pd.merge(self.datasets[dataset_name].data_1, self.datasets[dataset_name].data_2, how="left")
            self.datasets[dataset_name].combined_data["Mols"] = [Chem.MolFromSmiles(x) for x in list(self.datasets[dataset_name].combined_data["SMILES"])]
            self.datasets[dataset_name].combined_data.dropna(inplace=True, subset=["Mols"])
            self.datasets[dataset_name].combined_data.drop(["Mols"], inplace=True, axis=1)
            print("Datasets combined successfully.")
        else:
            print("Data not loaded. Please load both datasets before combining.")


    def split_datasets(
            self, 
            dataset_name: str):
        """
        Splits the combined dataset into training and validation sets.
        
        Args:
            dataset_name (str): The name of the dataset to split.
        
        Returns:
            training_set_final (pd.DataFrame) : The training set dataframe containing smiles, primary and confirmatory scores 
            validation_set (pd.DataFrame) : the validation set dataframe containing smiles, primary and confirmatory scores 
        """
        if self.datasets[dataset_name].combined_data is not None:
            #duplicates the dataframe to create one training and one validation set
            db_final = self.datasets[dataset_name].combined_data.copy()
            db_final_copy = self.datasets[dataset_name].combined_data.copy()

            #drop all non-confirmatory compounds 
            db_final_confirmatory = db_final_copy.dropna()

            #sort confirmatory samples according to primary screen score
            db_final_confirmatory_sorted = db_final_confirmatory.sort_values(by=['Score'], ascending=False)

            #chose top x as validation set
            percentage_split = self.validation_split
            validation_set_size = math.floor(len(db_final_confirmatory_sorted)*percentage_split)
            validation_set = db_final_confirmatory_sorted.head(validation_set_size)

            #delete validation compounds from training data set
            training_set = db_final_copy.drop(index=validation_set.index)

            #set training set indices to ensure continuous indices after validation set removal 
            training_set_reset = training_set.reset_index()
            training_set_final = training_set_reset.drop(training_set_reset.columns[0], axis=1)


            #recount training set primary and confimatory compounds and actives after train-val split
            n_train_primary = training_set_final.index.size
            act_train_primary = len([x for x in training_set_final["Primary"] if x == 1])

            n_train_confirmatory = len(training_set_final[~training_set_final['Confirmatory'].isnull()])
            act_train_confirmatory = np.sum(training_set_final["Confirmatory"])

            #recount validation set primary and confimatory compounds and actives after train-val split
            n_val_primary = validation_set.index.size
            act_val_primary = len([x for x in validation_set["Primary"] if x == 1])

            n_val_confirmatory = len(validation_set[~validation_set['Confirmatory'].isnull()])
            act_val_confirmatory = np.sum(validation_set["Confirmatory"])

            #test full dataset
            #current tests check that all confirmatory compounds were present before in the primary,
            #and that all duplicate records in either assay have been merged / removed
            db_primary = db_final.dropna(subset=["Primary"])
            assert len(db_final) == len(db_primary), "[cleanup]: All compounds in confirmatory should be present in primary assay"
            db_cut = db_final.drop_duplicates(subset=["SMILES"])
            assert len(db_final) == len(db_cut), "[cleanup]: There shouldn't be any more duplicate SMILES"
            print(f"[cleanup]: Tests passed successfully for full dataset")


            #test training set same as above
            train_primary = training_set_final.dropna(subset=["Primary"])
            assert len(training_set_final) == len(train_primary), "[cleanup]: All compounds in confirmatory should be present in primary assay"
            train_cut = training_set_final.drop_duplicates(subset=["SMILES"])
            assert len(training_set_final) == len(train_cut), "[cleanup]: There shouldn't be any more duplicate SMILES"
            print(f"[cleanup]: Tests passed successfully for training set")

            #test validation set same as above
            val_primary = validation_set.dropna(subset=["Primary"])
            assert len(validation_set) == len(val_primary), "[cleanup]: All compounds in confirmatory should be present in primary assay"
            val_cut = validation_set.drop_duplicates(subset=["SMILES"])
            assert len(validation_set) == len(val_cut), "[cleanup]: There shouldn't be any more duplicate SMILES"
            print(f"[cleanup]: Tests passed successfully for validation set")

            return training_set_final, validation_set
        else:
            logging.error("Data not loaded. Please load both datasets before combining.")

    def preprocess_data(self, path_to_raw: str):
        """
        Executes the complete preprocessing pipeline for all datasets given to the preprocessor.
        
        Args:
            None.
        
        Returns:
            None. Processes and updates each dataset in the dataset_names attribute.
        """
        for dataset_name in self.dataset_names:
            print("Preprocessing: " + dataset_name)
            #load the raw data downloaded from pubchem and stored in the Raw_Data folder
            self.load_data(dataset_name, path_to_raw)
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)
            #cleaning of primary and confirmatory datasets
            self.datasets[dataset_name].data_1 = self.clean_primary_data(self.datasets[dataset_name].data_1)
            self.datasets[dataset_name].data_2 = self.clean_confirmatory_data(self.datasets[dataset_name].data_2)
            lg.setLevel(RDLogger.INFO)
            #combines primary and confirmatory data to one datraframe
            self.combine_datasets(dataset_name)
            #splits combined dataframe into training and validation set
            self.datasets[dataset_name].training_set, self.datasets[dataset_name].validation_set= self.split_datasets(dataset_name)
            #saves activity labels as training and validation set labels in the repsective DatasetContainer
            self.datasets[dataset_name].training_set_labels = self.datasets[dataset_name].training_set_labels
            self.datasets[dataset_name].validation_set_labels = self.datasets[dataset_name].validation_set_labels
            print(f"Preprocessing completed successfully for {dataset_name}.")

    def save_data(
            self, 
            filename: str):
        """
        Saves the processed datasets to CSV files.
        
        Args:
            filename (str): The base path for saving the dataset files.
        
        Returns:
            None. Saves the training and validation datasets to CSV files.
        """
        if not os.path.exists(filename):
            os.mkdir(filename)
        for dataset_name in self.dataset_names:
            train_path = os.path.join(filename, f"{dataset_name}_train.csv")
            val_path = os.path.join(filename, f"{dataset_name}_val.csv")
            self.datasets[dataset_name].training_set.to_csv(train_path)
            self.datasets[dataset_name].validation_set.to_csv(val_path)
            print(f"[cleanup]: File saved for {dataset_name} at {filename}")



    def create_descriptors(
            self, 
            descriptor_type:str):
        """
        Generates molecular descriptors for the datasets. Options currently implemented are [smiles, ecfp, rdkit].
        You can add your own descriptor implementation below if you want to test other descriptors.
        
        Args:
            descriptor_type (str): The type of descriptor to generate ('smiles', 'ecfp', or 'rdkit').
        
        Returns:
            None. Updates the dataset containers with the generated descriptors.
        """
        if descriptor_type.lower() not in ["smiles", "ecfp", "rdkit"]:
            logging.error("Descriptor type not enabled. Please select one of the following: [smiles, ecfp, rdkit]")
        else:
            if descriptor_type.lower() == "smiles":
                #if smiles are used, the smiles from the training and validation set dataframes are set as the molecular descriptors
                for dataset_name in self.dataset_names:
                    self.datasets[dataset_name].training_set_descriptors = self.datasets[dataset_name].training_set["SMILES"]
                    self.validation_set_descriptors[dataset_name] = self.datasets[dataset_name].validation_set["SMILES"]
                    print(f"SMILES created successfully for {dataset_name}")
                    self.datasets[dataset_name].representation = "smiles"
            elif descriptor_type.lower() == "ecfp":
                #ECFPs are calculated and set in the datasetContainer as descriptors 
                for dataset_name in self.dataset_names:
                    mols_train = self.datasets[dataset_name].training_set["SMILES"]
                    mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
                    self.datasets[dataset_name].training_set_descriptors = get_ECFP(mols_train)

                    mols_val = self.datasets[dataset_name].validation_set["SMILES"]
                    mols_val = [Chem.MolFromSmiles(x) for x in mols_val]
                    self.datasets[dataset_name].validation_set_descriptors = get_ECFP(mols_val)
                    print(f"ECFPs created successfully for {dataset_name}")
                    self.datasets[dataset_name].representation = "ecfp"
            elif descriptor_type.lower() == "rdkit":
                #a collection of 208 RDKit descriptors are calculated and set in the datasetContainer as descriptors 
                for dataset_name in self.dataset_names:
                    scaler = StandardScaler()

                    mols_train = self.datasets[dataset_name].training_set["SMILES"]
                    mols_train = [Chem.MolFromSmiles(x) for x in mols_train]
                    rdkit_train =  get_rdkit_descr(mols_train)

                    mols_val = self.datasets[dataset_name].validation_set["SMILES"]
                    mols_val = [Chem.MolFromSmiles(x) for x in mols_val]
                    rdkit_val = get_rdkit_descr(mols_val)

                    scaler.fit(rdkit_train)
                    self.datasets[dataset_name].training_set_descriptors = scaler.transform(rdkit_train)
                    self.datasets[dataset_name].validation_set_descriptors = scaler.transform(rdkit_val)
                    self.datasets[dataset_name].representation = "rdkit"
                    print(f"RDkit Descriptors created successfully for {dataset_name}")

    def load_preprocessed_data(self, path_to_descriptors: str = None):
        """
        Loads preprocessed training and validation datasets from CSV files.
        
        Args:
            None.
        
        Returns:
            None. Updates the dataset containers with the loaded datasets.
        """
        if path_to_descriptors == None:
            path_to_descriptors = get_project_root()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.datasets:
                self.datasets[dataset_name] = self.DatasetContainer(dataset_name)
            training_path = path_to_descriptors / "Datasets" / dataset_name / (dataset_name + "_train.csv")
            self.datasets[dataset_name].training_set = pd.read_csv(training_path)
            validation_path = path_to_descriptors / "Datasets" / dataset_name / (dataset_name + "_val.csv")
            self.datasets[dataset_name].validation_set = pd.read_csv(validation_path)
            print(f"Processed datasets loaded successfully for {dataset_name}")
            self.datasets[dataset_name].training_set_labels = self.datasets[dataset_name].training_set["Primary"]
            self.datasets[dataset_name].validation_set_labels = self.datasets[dataset_name].validation_set["Confirmatory"]
            
            
            
    def load_precalculated_descriptors(
            self, 
            representation: str):
        """
        Loads precalculated molecular descriptors from files.
        
        Args:
            representation(str): The type of descriptor representation to load (here it only matters whether its smiles or not.
                                                                                if it isnt smiles, the representation given here should just match
                                                                                the name given the file names used for importing - see down below).
        
        Returns:
            None. Updates the dataset containers with the loaded descriptors.
        """

        if representation.lower() == "smiles":
            for dataset_name in self.dataset_names:
                if dataset_name not in self.datasets:
                    self.datasets[dataset_name] = self.DatasetContainer(dataset_name)

                self.datasets[dataset_name].training_set_descriptors = self.datasets[dataset_name].training_set["SMILES"]
                self.datasets[dataset_name].validation_set_descriptors = self.datasets[dataset_name].validation_set["SMILES"]
                self.datasets[dataset_name].representation = "smiles"
                print(f"SMILES loaded successfully for {dataset_name}")
        else:
            for dataset_name in self.dataset_names:
                if dataset_name not in self.datasets:
                    self.datasets[dataset_name] = self.DatasetContainer(dataset_name)

                train_path = "../Datasets_descr/" + dataset_name + "/" + dataset_name + "_" + representation + "_train.pkl"
                val_path = "../Datasets_descr/" + dataset_name + "/" + dataset_name + "_" + representation + "_val.pkl"


                x_train = pd.read_pickle(train_path)
                x_train = x_train.to_numpy()

                x_val = pd.read_pickle(val_path)
                x_val = x_val.to_numpy()
                self.datasets[dataset_name].training_set_descriptors = x_train
                self.datasets[dataset_name].validation_set_descriptors = x_val
                self.datasets[dataset_name].representation = representation
                print(f"{representation} loaded successfully for {dataset_name}")

    def get_dataset(self, dataset_name):
        try:
            return self.datasets[dataset_name]
        except KeyError:
            # Handle the KeyError exception
            available_names = ', '.join(self.dataset_names)
            print(f"Use one of the following names: {available_names}")
            return
            
