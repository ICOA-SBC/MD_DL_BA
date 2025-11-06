# Script to generate .npy files for use with spatio-temporal models
# disclamer: the github version of this file wasn't properly tested. You will likely need to debug it.

# create folders and files in them containing information about MD simulations
# prior to executing this script, you need to:
# 1- extract frames from the MD simulations
# 2- calculate the pocket (training on whole complex takes too long, using 12A around the ligand seems a good compromise)
# 3- calculate .mol2 files in a similar way to https://gitlab.com/cheminfIBB/pafnucy/-/blob/master/pdbbind_data.ipynb?ref_type=heads
# affinity data will used directly during training/inference from INDEX_general_PL_data.2019 and INDEX_refined_data.2019 (downloadable from PDBbind)


import numpy as np
import pandas as pd
import os
import glob
import pathlib
import argparse
import ast

import pybel
from tfbio.data import Featurizer

parser = argparse.ArgumentParser(
    description='Generate .npy files using frames extracted from MD simulation. These .npy files are compatible with using spatio-temporal models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--path_MD', '-p', required=True,
                    help='Path to the folder where are stored the frames extracted from MD trajectories (the frames have to be stored in subfolders containing the pdb name)')
parser.add_argument('--output', '-o', required=True,
                    help='Path to where will be created the datasets for DL')
parser.add_argument('--split_data', '-s', required=True,
                    help='provide a .csv to determine how the data should be split. Generate it via https://gitlab.com/cheminfIBB/pafnucy/-/blob/master/pdbbind_data.ipynb?ref_type=heads. We also added a column to know if we did MD simulation on the PDB.')
parser.add_argument('--number_replicates', '-n', default="['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']",
                    help='give a list of the MD simulation replicate numbers used for each pdb')
parser.add_argument('--frame_number', '-f', default=50,
                    help='provide the number of frames extracted per simulation')
parser.add_argument('--data_type', '-d', default='complex',
                    help='do we want to train/evaluate model on whole complex, or solely on a part of it. choice between "complex", "protein" or "ligand"')
parser.add_argument('--tracking_ligand', '-t', default='False',
                    help='using only ligand, you might want to track ')
parser.add_argument('--redo_all', '-r', default='False',
                    help='force the recreation of .npy files that were already created')
parser.add_argument('--only_counting', '-oc', default='False',
                    help='do not create datasets and return the number of complexes and simulations per datasets')
args = parser.parse_args()

path_MD = args.path_MD
path_output = args.output
split_data = pd.read_csv(args.split_data)

num_replicates = ast.literal_eval(args.number_replicates)
frame_number_total = args.frame_number

data_type = args.data_type #could be 'protein' or 'ligand' or 'complex'
if data_type == 'complex':
    name_dataset = f'4D_data_{data_type}'
else:
    name_dataset = f'4D_data_only_{data_type}'

tracking_ligand = ast.literal_eval(args.tracking_ligand)
redo_all = ast.literal_eval(args.redo_all)
only_counting = ast.literal_eval(args.only_counting)


featurizer = Featurizer()
charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

counter_general = 0
counter_refined = 0
counter_core = 0
counter_core2013 = 0
counter_core2013_simulations = 0
counter_general_simulations = 0
counter_refined_simulations = 0
counter_core_simulations = 0

# Lwrong_mol2 = []

path_data_dl = f'{path_output}/{name_dataset}'
pathlib.Path(path_data_dl).mkdir(parents=True, exist_ok=True)
datasets = os.listdir(path_data_dl)
Ddone_pdb = {}
for dataset in datasets: #ie general refined core core2013
    simulation_pdb_done = 0
    num=0
    for num,pdb in enumerate(os.listdir(f'{path_data_dl}/{dataset}'),1):
        Ddone_pdb[pdb] = [i.strip('.npy').split('_')[1] for i in os.listdir(f'{path_data_dl}/{dataset}/{pdb}')]
        simulation_pdb_done += len(os.listdir(f'{path_data_dl}/{dataset}/{pdb}'))
    print(f'{dataset}: {num} complexes and {simulation_pdb_done} simulations')
if only_counting:
    raise

for _, row in split_data.iterrows():
    pdb = row['pdbid']
    md = row['MD'] #True or False, to know on which pdb of the PDBbind we carried simulations
    if not md:
        continue
    include = row['include']
    dataset = row['set']

    if not redo_all:
        if pdb in Ddone_pdb:
            if len(Ddone_pdb[pdb]) == len(num_replicates): #improvement to do: should compare the replicate numbers instead of looking at length of each list
                continue
            else:
                to_do = list(set(num_replicates) - set(Ddone_pdb[pdb]))
                to_do.sort()
        else:
            to_do = num_replicates
    else:
        to_do = num_replicates

    flag_several_MD = False
    flag_count = False
    for replicate_number in to_do: #assess the number of frames per simulations
        Lmd_pockets = [int(os.path.basename(i).replace('pocket_','')) for i in glob.glob(f'{path_MD}/{pdb}/replicate_{replicate_number}/pocket_*.mol2')]
        if len(Lmd_pockets) != frame_number_total: #in the case of spatiotemporal learning we keep only full simulations (ie, 50 frames in our case, modify that number to the number of frames you extracted per replicate)
            print(f'no {frame_number_total} frames simulations: {pdb} | replicate {replicate_number}')
            continue

        LMD_data_DL = []
        Lmd_pockets.sort()
        flag_fail = False
        for frame_number in Lmd_pockets:
            if data_type == 'complex' or data_type == 'ligand': #ignore ligand when using only protein
                try:
                    ligand = next(pybel.readfile('mol2', f'{path_MD}/{pdb}/replicate_{replicate_number}/ligand_{frame_number}.mol2'))
                    # do not add the hydrogens! they are in the structure and it would reset the charges
                except:
                    print(f'no ligand for {pdb} replicate {replicate_number} frame number {frame_number}')
                    flag_fail = True
                    break
                ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
                assert (ligand_features[:, charge_idx] != 0).any()
            if data_type == 'complex' or data_type == 'protein':
                if not tracking_ligand: #doesn't execute in case we don't want to provide empty data when ligand is out of pocket with only ligand
                    try:
                        pocket = next(pybel.readfile('mol2', f'{path_MD}/{pdb}/replicate_{replicate_number}/pocket_{frame_number}.mol2')) #using pdb, we don't get the same charges. But the pdb to mol2 transition can lead to losing some frames
                        # do not add the hydrogens! they were already added in chimera and it would reset the charges
                    except:
                        print(f'no pocket for {pdb} replicate {replicate_number} frame number {frame_number}')
                        flag_fail = True
                        break #break instead of continue because we need all frames for MD Conv LSTM learning (in our case 50 frames)
                    pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
                    assert (pocket_features[:, charge_idx] != 0).any()

            if data_type == 'protein':
                centroid = pocket_coords.mean(axis=0)
                pocket_coords -= centroid
                data_DL = np.concatenate((pocket_coords,pocket_features),axis=1)
            else:
                if data_type == 'ligand' and tracking_ligand: #in case we want do not want to provide empty data when ligand is out of pocket with only ligand
                    centroid = ligand_coords.mean(axis=0)
                    ligand_coords -= centroid
                    data_DL = np.concatenate((ligand_coords,ligand_features),axis=1)
                else:
                    centroid = pocket_coords.mean(axis=0) #Without tracking: we use pocket as the centroid, since with frames you can have ligand out of the binding site
                    ligand_coords -= centroid
                    pocket_coords -= centroid
                    if data_type == 'complex':
                        data_DL = np.concatenate((np.concatenate((ligand_coords, pocket_coords)),
                                    np.concatenate((ligand_features, pocket_features))), axis=1)
                    elif data_type == 'ligand':
                        data_DL = np.concatenate((ligand_coords,ligand_features),axis=1)
            
            # print(data_DL.shape)
            LMD_data_DL.append(data_DL)

        if flag_fail:
            continue

        MD_data_DL = np.array([np.array(xi) for xi in LMD_data_DL])
        print(MD_data_DL.shape)
        
        if MD_data_DL.shape[0] != frame_number_total: 
            continue

        # if data_type == 'complex' or data_type == 'protein':
        #     if len(MD_data_DL.shape) > 1:
        #         print(f'wrong PDB please investigate {pdb} replicate {replicate_number}') #can be same frame as been printed several times in a row, thus giving the same shape on the frames vector (50 in our case)
        #         Lwrong_mol2.append(pdb)
        #         continue
        # elif data_type == 'ligand':
        #     if len(MD_data_DL.shape) != 3:
        #         print(f'wrong ligand please investigate {pdb} replicate {replicate_number}') #check if the ligands isn't modified during the simulation (spot any possible issue)
        #         Lwrong_mol2.append(pdb)
        #         continue
        

        if not include:
            pathlib.Path(f"{path_data_dl}/core2013/{pdb}").mkdir(parents=True, exist_ok=True)
            np.save(f"{path_data_dl}/core2013/{pdb}/replicate_{replicate_number}.npy", MD_data_DL)
            counter_core2013_simulations += 1
        elif dataset == 'general':
            pathlib.Path(f"{path_data_dl}/general/{pdb}").mkdir(parents=True, exist_ok=True)
            np.save(f"{path_data_dl}/general/{pdb}/replicate_{replicate_number}.npy", MD_data_DL)
            counter_general_simulations += 1
        elif dataset == 'refined':
            pathlib.Path(f"{path_data_dl}/refined/{pdb}").mkdir(parents=True, exist_ok=True)
            np.save(f"{path_data_dl}/refined/{pdb}/replicate_{replicate_number}.npy", MD_data_DL)
            counter_refined_simulations += 1
        elif dataset == 'core':
            pathlib.Path(f"{path_data_dl}/core/{pdb}").mkdir(parents=True, exist_ok=True)
            np.save(f"{path_data_dl}/core/{pdb}/replicate_{replicate_number}.npy", MD_data_DL)
            counter_core_simulations += 1
        print(pdb, f'replicate {replicate_number}')
        if not flag_count:
            flag_count = True
        break
        
    if flag_count:
        if not include:
            counter_core2013 += 1
        elif dataset == 'general':
            counter_general += 1
        elif dataset == 'refined':
            counter_refined += 1
        elif dataset == 'core':
            counter_core += 1
 
print(f'general: prepared {counter_general} complexes and {counter_general_simulations} simulations')
print(f'refined: prepared {counter_refined} complexes and {counter_refined_simulations} simulations')
print(f'core: prepared {counter_core} complexes and {counter_core_simulations} simulations')
print(f'core2013: prepared {counter_core2013} complexes and {counter_core2013_simulations} simulations')

# print('issues with those mol2 files:',set(Lwrong_mol2))

