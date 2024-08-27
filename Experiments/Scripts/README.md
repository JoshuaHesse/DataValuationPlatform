## Content
This folder contains primary cleanup and descriptor calculation pipelines as well as the subfolders for the application-specific scripts. Additionally, if you want to reproduce the Undersampling results, you will have to install the [Moldata benchmark git repository](https://github.com/LumosBio/MolData) in this folder.
### Folders
- [ActiveLearning:](ActiveLearning) contains all scripts for the active learning application
- [FalsePositivePrediction:](FalsePositivePrediction) contains all scripts for the false positive prediction application
- [Undersampling:](Undersampling) contains all scripts for the undersampling project
 
### Scripts
- [cleanup_pipeline_jh.py:](cleanup_pipeline_jh.py) Executes the standardization and assay merging script to generate datasets ready to be
analyzed. Outputs are saved in [Datasets](../Datasets), Outputs are saved in [Logs](../Logs).
- [descr_export_pipeline_jh.py:](descr_export_pipeline_jh) Uses the preprocessed datasets to calculate and save the molecular desriptors of choice ( ECFPs, RDKit descriptors)
which are then loaded directly during the application scripts instead of having to recalculate them every time. Results are saved in [Datasets_descr](../Datasets_descr).
- [mol_data_descr_export_pipeline_jh.py:](descr_export_pipeline_jh) Uses the moldata benchmark to calculate and save the molecular desriptors of choice ( ECFPs, SMILES, RDKit descriptors)
which are then loaded directly during the application scripts instead of having to recalculate them every time. To use this, please first clone the [Moldata benchmark git repository](https://github.com/LumosBio/MolData) in the current folder. Results are saved in [Datasets_descr_MolData](../Datasets_descr_MolData).

### Example
Here is a tutorial of how to use the scripts provided here:
Begin by using the [cleanup_pipeline_jh.py:](cleanup_pipeline_jh.py) to create the dataset files:
```
cd /DataValuationPlatform/Experiments/Scripts
python3 cleanup_pipeline_jh.py --AID_1 example_1 --AID_2 example_2 --filename new_dataset --percentage 10
```
This script cleans the data and creates the train and validation sets in the [Datasets](../Datasets) folder. The AIDs and correspondent names can be found in the [Metadata](../Logs/dataset_metadata.csv) file.
After the dataset has been processed, use the [descr_export_pipeline_jh.py:](descr_export_pipeline_jh) to create the descriptors of your choice, here ECFP:
```
python3 descr_export_pipeline_jh.py --dataset new_dataset --representation ECFP
```
The script creates descriptors for both train and validation set and saves them in [Datasets_descr](../Datasets_descr)

Now you can start with the applications - their usages are explained in the respective folders.
