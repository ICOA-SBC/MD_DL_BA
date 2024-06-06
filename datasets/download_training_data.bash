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
unzip 4D_data/data.zip -d 4D_data
rm 4D_data/data.zip

wget https://zenodo.org/record/10390550/files/4D_complex_validation_set.zip?download=1 -O 4D_data/data.zip
unzip 4D_data/data.zip -d 4D_data
rm 4D_data/data.zip

wget https://zenodo.org/record/10390550/files/4D_only_ligand_training_set.zip?download=1 -O 4D_data_only_ligand/data.zip
unzip 4D_data_only_ligand/data.zip -d 4D_data_only_ligand
rm 4D_data_only_ligand/data.zip

wget https://zenodo.org/record/10390550/files/4D_only_ligand_validation_set.zip?download=1 -O 4D_data_only_ligand/data.zip
unzip 4D_data_only_ligand/data.zip -d 4D_data_only_ligand
rm 4D_data_only_ligand/data.zip

wget https://zenodo.org/record/10390550/files/4D_only_ligand_tracking_training_set.zip?download=1 -O 4D_data_only_ligand_tracking/data.zip
unzip 4D_data_only_ligand_tracking/data.zip -d 4D_data_only_ligand_tracking
rm 4D_data_only_ligand_tracking/data.zip

wget https://zenodo.org/record/10390550/files/4D_only_ligand_tracking_validation_set.zip?download=1 -O 4D_data_only_ligand_tracking/data.zip
unzip 4D_data_only_ligand_tracking/data.zip -d 4D_data_only_ligand_tracking
rm 4D_data_only_ligand_tracking/data.zip

wget https://zenodo.org/record/10390550/files/4D_only_protein_training_set.zip?download=1 -O 4D_data_only_protein/data.zip
unzip 4D_data_only_protein/data.zip -d 4D_data_only_protein
rm 4D_data_only_protein/data.zip

wget https://zenodo.org/record/10390550/files/4D_only_protein_validation_set.zip?download=1 -O 4D_data_only_protein/data.zip
unzip 4D_data_only_protein/data.zip -d 4D_data_only_protein
rm 4D_data_only_protein/data.zip

wget https://zenodo.org/record/10390550/files/initial_training_set.zip?download=1 -O no_MD/data.zip
unzip no_MD/data.zip -d no_MD
rm no_MD/data.zip

wget https://zenodo.org/record/10390550/files/initial_validation_set.zip?download=1 -O no_MD/data.zip
unzip no_MD/data.zip -d no_MD
rm no_MD/data.zip

wget https://zenodo.org/record/10390550/files/MDDA_complex_training_set_augmented.zip?download=1 -O MD_DA/complex/data.zip
unzip MD_DA/complex/data.zip -d MD_DA/complex
rm MD_DA/complex/data.zip

wget https://zenodo.org/record/10390550/files/MDDA_complex_validation_set_augmented.zip?download=1 -O MD_DA/complex/data.zip
unzip MD_DA/complex/data.zip -d MD_DA/complex
rm MD_DA/complex/data.zip

wget https://zenodo.org/record/10390550/files/MDDA_only_ligand_training_set_augmented.zip?download=1 -O MD_DA/only_ligand/data.zip
unzip MD_DA/only_ligand/data.zip -d MD_DA/only_ligand
rm MD_DA/only_ligand/data.zip

wget https://zenodo.org/record/10390550/files/MDDA_only_ligand_validation_set_augmented.zip?download=1 -O MD_DA/only_ligand/data.zip
unzip MD_DA/only_ligand/data.zip -d MD_DA/only_ligand
rm MD_DA/only_ligand/data.zip

wget https://zenodo.org/record/10390550/files/MDDA_only_protein_training_set_augmented.zip?download=1 -O MD_DA/only_protein/data.zip
unzip MD_DA/only_protein/data.zip -d MD_DA/only_protein
rm MD_DA/only_protein/data.zip

wget https://zenodo.org/record/10390550/files/MDDA_only_protein_validation_set_augmented.zip?download=1 -O MD_DA/only_protein/data.zip
unzip MD_DA/only_protein/data.zip -d MD_DA/only_protein
rm MD_DA/only_protein/data.zip

wget https://zenodo.org/record/10390550/files/reduced_training_set.zip?download=1 -O reduced/data.zip
unzip reduced/data.zip -d reduced
rm reduced/data.zip

wget https://zenodo.org/record/10390550/files/reduced_validation_set.zip?download=1 -O reduced/data.zip
unzip reduced/data.zip -d reduced
rm reduced/data.zip

echo "Downloaded and unpacked training/validation sets"
