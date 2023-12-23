#!/bin/bash

mkdir -p 4D_data
mkdir -p 4D_data_only_ligand
mkdir -p 4D_data_only_ligand_tracking
mkdir -p 4D_data_only_protein
mkdir -p no_MD
mkdir -p MD_DA/complex
mkdir -p MD_DA/only_ligand
mkdir -p MD_DA/only_protein
mkdir -p reduced

wget https://zenodo.org/record/10390550/files/4D_complex_training_set.zip?download=1 -O 4D_data/data.zip
unzip 4D_data/data.zip 4D_data/training_set

wget https://zenodo.org/record/10390550/files/4D_complex_validation_set.zip?download=1 -O 4D_data/data.zip
unzip 4D_data/data.zip 4D_data/validation_set

wget https://zenodo.org/record/10390550/files/4D_only_ligand_training_set.zip?download=1 -O 4D_data_only_ligand/data.zip
unzip 4D_data_only_ligand/data.zip 4D_data_only_ligand/training_set

wget https://zenodo.org/record/10390550/files/4D_only_ligand_validation_set.zip?download=1 -O 4D_data_only_ligand/data.zip
unzip 4D_data_only_ligand/data.zip 4D_data_only_ligand/validation_set

wget https://zenodo.org/record/10390550/files/4D_only_ligand_tracking_training_set.zip?download=1 -O 4D_data_only_ligand_tracking/data.zip
unzip 4D_data_only_ligand_tracking/data.zip 4D_data_only_ligand_tracking/training_set

wget https://zenodo.org/record/10390550/files/4D_only_ligand_tracking_validation_set.zip?download=1 -O 4D_data_only_ligand_tracking/data.zip
unzip 4D_data_only_ligand_tracking/data.zip 4D_data_only_ligand_tracking/validation_set

wget https://zenodo.org/record/10390550/files/4D_only_protein_training_set.zip?download=1 -O 4D_data_only_protein/data.zip
unzip 4D_data_only_protein/data.zip 4D_data_only_protein/training_set

wget https://zenodo.org/record/10390550/files/4D_only_protein_validation_set.zip?download=1 -O 4D_data_only_protein/data.zip
unzip 4D_data_only_protein/data.zip 4D_data_only_protein/validation_set

wget https://zenodo.org/record/10390550/files/initial_training_set.zip?download=1 -O no_MD/data.zip
unzip no_MD/data.zip no_MD/training_set.hdf

wget https://zenodo.org/record/10390550/files/initial_validation_set.zip?download=1 -O no_MD/data.zip
unzip no_MD/data.zip no_MD/validation_set.hdf

wget https://zenodo.org/record/10390550/files/MDDA_complex_training_set_augmented.zip?download=1 -O MD_DA/complex/data.zip
unzip MD_DA/complex/data.zip MD_DA/complex/training_set.hdf

wget https://zenodo.org/record/10390550/files/MDDA_complex_validation_set_augmented.zip?download=1 -O MD_DA/complex/data.zip
unzip MD_DA/complex/data.zip MD_DA/complex/validation_set.hdf

wget https://zenodo.org/record/10390550/files/MDDA_only_ligand_training_set_augmented.zip?download=1 -O MD_DA/only_ligand/data.zip
unzip MD_DA/only_ligand/data.zip MD_DA/only_ligand/training_set.hdf

wget https://zenodo.org/record/10390550/files/MDDA_only_ligand_validation_set_augmented.zip?download=1 -O MD_DA/only_ligand/data.zip
unzip MD_DA/only_ligand/data.zip MD_DA/only_ligand/validation_set.hdf

wget https://zenodo.org/record/10390550/files/MDDA_only_protein_training_set_augmented.zip?download=1 -O MD_DA/only_protein/data.zip
unzip MD_DA/only_protein/data.zip MD_DA/only_protein/training_set.hdf

wget https://zenodo.org/record/10390550/files/MDDA_only_protein_validation_set_augmented.zip?download=1 -O MD_DA/only_protein/data.zip
unzip MD_DA/only_protein/data.zip MD_DA/only_protein/validation_set.hdf

wget https://zenodo.org/record/10390550/files/reduced_training_set.zip?download=1 -O reduced/data.zip
unzip reduced/data.zip reduced/training_set.hdf

wget https://zenodo.org/record/10390550/files/reduced_validation_set.zip?download=1 -O reduced/data.zip
unzip reduced/data.zip reduced/validation_set.hdf

echo "Downloaded and unpacked training/validation sets"