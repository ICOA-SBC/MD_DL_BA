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
mkdir -p fep

wget https://zenodo.org/record/10390550/files/4D_complex_test_set.zip?download=1 -O 4D_data/data.zip
unzip 4D_data/data.zip 4D_data/test_set

wget https://zenodo.org/record/10390550/files/4D_only_ligand_test_set.zip?download=1 -O 4D_data_only_ligand/data.zip
unzip 4D_data_only_ligand/data.zip 4D_data_only_ligand/test_set

wget https://zenodo.org/record/10390550/files/4D_only_ligand_tracking_test_set.zip?download=1 -O 4D_data_only_ligand_tracking/data.zip
unzip 4D_data_only_ligand_tracking/data.zip 4D_data_only_ligand_tracking/test_set

wget https://zenodo.org/record/10390550/files/4D_only_protein_test_set.zip?download=1 -O 4D_data_only_protein/data.zip
unzip 4D_data_only_protein/data.zip 4D_data_only_protein/test_set

wget https://zenodo.org/record/10390550/files/initial_test_set.zip?download=1 -O no_MD/data.zip
unzip no_MD/data.zip no_MD/test_set.hdf

wget https://zenodo.org/record/10390550/files/MDDA_complex_test_set_augmented.zip?download=1 -O MD_DA/complex/data.zip
unzip MD_DA/complex/data.zip MD_DA/complex/test_set.hdf

wget https://zenodo.org/record/10390550/files/MDDA_only_ligand_test_set_augmented.zip?download=1 -O MD_DA/only_ligand/data.zip
unzip MD_DA/only_ligand/data.zip MD_DA/only_ligand/test_set.hdf

wget https://zenodo.org/record/10390550/files/MDDA_only_protein_test_set_augmented.zip?download=1 -O MD_DA/only_protein/data.zip
unzip MD_DA/only_protein/data.zip MD_DA/only_protein/test_set.hdf

wget https://zenodo.org/record/10390550/files/reduced_test_set.zip?download=1 -O reduced/data.zip
unzip reduced/data.zip reduced/test_set.hdf

wget https://zenodo.org/record/10390550/files/fep_test_set.zip?download=1 -O fep/data.zip
unzip fep/data.zip fep/fep.hdf

echo "Downloaded and unpacked test sets"