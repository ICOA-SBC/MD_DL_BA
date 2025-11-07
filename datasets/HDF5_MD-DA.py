# Script to generate hdf5 datasets used to train DL models
# disclamer: the github version of this file wasn't properly tested. You will likely need to debug it.

# create HDF5 files containing information about MD simulations for data augmentation
# this script can be run to prepare dataset from solely the PDBbind or also from MD simulations
# In the case you use MD simulations, and prior to executing this script, you need to:
# 1- extract frames from the MD simulations
# 2- calculate the pocket (training on whole complex takes too long, using 12A around the ligand seems a good compromise)
# 3- calculate .mol2 files in a similar way to https://gitlab.com/cheminfIBB/pafnucy/-/blob/master/pdbbind_data.ipynb?ref_type=heads
# for affinity data, the .csv file can be calculated in this way https://gitlab.com/cheminfIBB/pafnucy/-/blob/master/pdbbind_data.ipynb, we also added the information on which pdb we performed simulation in a new column

import numpy as np
import pandas as pd
import h5py
import os
import glob
import argparse
import pathlib
import ast

import pybel
from tfbio.data import Featurizer


parser = argparse.ArgumentParser(
    description='Generate hdf5 dataset using frames extracted from MD simulation to perform data augmentation for training/evaluating DL models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--path_MD', '-p', required=True,
                    help='Path to the folder where are stored the frames extracted from MD trajectories (the frames have to be stored in subfolders containing the pdb name)')
parser.add_argument('--path_pdbbind', '-pp', default='',
                    help='Path of the pdbbind to use the cristallographic structures. Not necessary, but we performed data-augmentation, so MD frames are used in combination to PDBbind structures.')
parser.add_argument('--output', '-o', required=True,
                    help='Path to where will be created the datasets for DL')
parser.add_argument('--split_data', '-s', required=True,
                    help='provide a .csv to determine how the data should be split. Generate it via https://gitlab.com/cheminfIBB/pafnucy/-/blob/master/pdbbind_data.ipynb?ref_type=heads. We also added a column to know if we did MD simulation on the PDB.')
parser.add_argument('--pocket_type', '-pt', default='pocket',
                    help='name of the pocket files. Can be used to train on whole protein, but computation time is way more important (for the PDBbind: 20 min on pockets with a size of 12A from ligand, 5h on protein)')
parser.add_argument('--method', '-m', default='all',
                    help='use min_max or every_10, if you want to include in the dataset only specific frames like the first and last, or 1 frame every 10 etc')
parser.add_argument('--number_replicates', '-n', default="['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']",
                    help='give a list of the MD simulation replicate numbers used for each pdb')
# parser.add_argument('--frame_number', '-f', default=50,
#                     help='provide the number of frames extracted per simulation')
parser.add_argument('--data_type', '-d', default='complex',
                    help='do we want to train/evaluate model on whole complex, or solely on a part of it. choice between "complex", "protein" or "ligand"')
parser.add_argument('--tracking_ligand', '-t', default='False',
                    help='using only ligand, you might want to track ')
parser.add_argument('--only_counting', '-oc', default='False',
                    help='do not create datasets and return the number of complexes and simulations per datasets')
args = parser.parse_args()

path_MD = args.path_MD
path_pdbbind = args.path_pdbbind
path_output = args.output
split_data = pd.read_csv(args.split_data)

method = args.method
num_replicates = ast.literal_eval(args.number_replicates)
# frame_number_total = args.frame_number

data_type = args.data_type #could be 'protein' or 'ligand' or 'complex'
if data_type == 'complex':
    name_dataset = f'data_augmentation_{data_type}'
else:
    name_dataset = f'data_augmentation_only_{data_type}'
pocket_type = args.pocket_type

tracking_ligand = ast.literal_eval(args.tracking_ligand)
only_counting = ast.literal_eval(args.only_counting)

path_data_dl = f'{path_output}/datasets'
pathlib.Path(path_data_dl).mkdir(parents=True, exist_ok=True)

dataset_path = {'general': 'general-set-except-refined', 'refined': 'refined-set', 'core': 'refined-set'}

# MD_hdf5 = False #Old code used to create hdf5 dataset for training spatiotemporal data. Led to issues like OOM during training.



featurizer = Featurizer()
charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

if only_counting:
    for dataset_name in ['core','core2013','refined','general']:
        with h5py.File(f'{path_data_dl}/{dataset_name}.hdf', 'r') as f:
            Lframes = []
            Lsims = []
            Lcomplexes = []
            for pdb_id in f:
                Lpdb_id = pdb_id.split('_')
                Lframes.append(pdb_id)
                if len(Lpdb_id) > 1:
                    Lsims.append(Lpdb_id[0]+Lpdb_id[1])
                    Lcomplexes.append(Lpdb_id[0])
            print(f'{dataset_name}: {len(set(Lcomplexes))} complexes, {len(set(Lsims))} simulations, {len(set(Lframes))} frames')
    raise

Dsum_up = {}
# flag = False
with h5py.File(f'{path_data_dl}/core2013.hdf', 'w') as g:
    core2013_counter = 0

    for dataset_name, data in split_data.groupby('set'): 
        ds_path = dataset_path[dataset_name] 
        print(dataset_name, 'set')
        counter = 0
        with h5py.File(f'{path_data_dl}/{dataset_name}.hdf', 'w') as f: 
            for _, row in data.iterrows():
                pdb = row['pdbid']
                affinity = row['-logKd/Ki']
                md = row['MD']

                # if path_pdbbind and not MD_hdf5:
                if path_pdbbind:
                    print(pdb)
                    if dataset_name == 'core': #the is core complexes in both refined and general data sets
                        ds_path = 'refined-set'
                        if not os.path.exists(f'{path_pdbbind}/{ds_path}/{pdb}/{pdb}_ligand.mol2'):
                            ds_path = 'general-set-except-refined'
                    if data_type == 'complex' or data_type == 'ligand':
                        try:
                            ligand = next(pybel.readfile('mol2', f'{path_pdbbind}/{ds_path}/{pdb}/{pdb}_ligand.mol2'))
                        except:
                            print(f'no ligand for {pdb} (not MD)')
                            continue
                        ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1) # comment when doing prot only
                        assert (ligand_features[:, charge_idx] != 0).any() # comment when doing prot only
                        centroid = ligand_coords.mean(axis=0)
                        ligand_coords -= centroid

                    if data_type == 'complex' or data_type == 'protein':
                        try:
                            pocket = next(pybel.readfile('mol2', f'{path_pdbbind}/{ds_path}/{pdb}/{pdb}_{pocket_type}.mol2'))
                            # do not add the hydrogens! they were already added in chimera and it would reset the charges
                        except:
                            print(f'no pocket for {pdb} (not MD)')
                            continue
                        pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1) # comment when doing ligand only
                        assert (pocket_features[:, charge_idx] != 0).any() # comment when doing ligand only
                        if data_type == 'protein':
                            centroid = pocket_coords.mean(axis=0)
                        pocket_coords -= centroid

                    if data_type == 'complex':
                        data_DL = np.concatenate((np.concatenate((ligand_coords, pocket_coords)),
                                        np.concatenate((ligand_features, pocket_features))), axis=1)
                    elif data_type == 'ligand':
                        data_DL = np.concatenate((ligand_coords,ligand_features),axis=1)
                    elif data_type == 'protein':
                        data_DL = np.concatenate((pocket_coords,pocket_features),axis=1)

            
                    if row['include']:
                        dataset = f.create_dataset(f'{pdb}', data=data_DL, shape=data_DL.shape, dtype='float32', compression='lzf')
                        dataset.attrs['affinity'] = affinity
                        counter += 1
                    else:
                        dataset = g.create_dataset(f'{pdb}', data=data_DL, shape=data_DL.shape, dtype='float32', compression='lzf')
                        dataset.attrs['affinity'] = affinity
                        core2013_counter += 1
                
                if not md:
                    continue
                else:
                    # if not path_pdbbind or MD_hdf5:
                    if not path_pdbbind:
                        print(pdb)

                print('MD frames')

                flag_count = False
                for replicate_number in num_replicates: #look the number of frames per simulations
                    Lmd_pockets = [int(os.path.basename(i).replace(f'{pocket_type}_','')) for i in glob.glob(f'{path_MD}/{pdb}/replicate_{replicate_number}/{pocket_type}_*.mol2')]

                    if path_pdbbind: #remove first frame, since we get the cristallo one
                        if 1 in Lmd_pockets:
                            Lmd_pockets.remove(1)
                    # if MD_hdf5: #we prefer to use files into folder type of dataset for spatio-temporal learning
                    #     if len(Lmd_pockets) != frame_number_total:
                    #         print(f"{pdb} : replicate {replicate_number} not included, doesn't have all pockets (50 in our case)")
                    #         continue
                    #     LMD_data_DL = []
                    if not Lmd_pockets:
                        print(f'no pocket for replicate {replicate_number}')
                        continue
                    if len(Lmd_pockets) < 1:
                        print(f'no pocket for replicate {replicate_number}')
                        continue

                    Lmd_pockets.sort()
                    
                    # these are the extraction types: if we want all the frames or only specific ones
                    # if MD_hdf5 or method == 'all':
                    if method == 'all':
                        Dframe_numbers = {pocket_number:pocket_number for pocket_number in Lmd_pockets}
                    elif method == 'min_max':
                        Dframe_numbers = {'min': min(Lmd_pockets),'max': max(Lmd_pockets)}
                    elif method == 'every_10':
                        Dframe_numbers = {pocket_number:pocket_number for pocket_number in Lmd_pockets if pocket_number == 1 or not pocket_number%10}

                    print(f'replicate {replicate_number}')
                    for name,frame_number in Dframe_numbers.items():
                        if data_type == 'complex' or data_type == 'ligand':
                            try:
                                ligand = next(pybel.readfile('mol2', f'{path_MD}/{pdb}/replicate_{replicate_number}/ligand_{frame_number}.mol2'))
                                # do not add the hydrogens! they are in the structure and it would reset the charges
                            except:
                                print(f'no ligand for {pdb} replicate {replicate_number} frame number {frame_number}')
                                continue
                            ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
                            assert (ligand_features[:, charge_idx] != 0).any()

                        if data_type == 'complex' or data_type == 'protein' or tracking_ligand: #in case we want do not want to provide empty data when ligand is out of pocket with only ligand
                            try:
                                pocket = next(pybel.readfile('mol2', f'{path_MD}/{pdb}/replicate_{replicate_number}/{pocket_type}_{frame_number}_static.mol2'))
                                # do not add the hydrogens! they were already added in chimera and it would reset the charges
                            except:
                                print(f'no pocket for {pdb} replicate {replicate_number} frame number {frame_number}')
                                continue
                            pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
                            assert (pocket_features[:, charge_idx] != 0).any()

                        if data_type == 'protein':
                            centroid = pocket_coords.mean(axis=0)
                            pocket_coords -= centroid
                            data_DL = np.concatenate((pocket_coords,pocket_features),axis=1)
                        elif tracking_ligand and data_type == 'ligand': #in case we want do not want to provide empty data when ligand is out of pocket with only ligand
                            centroid = ligand_coords.mean(axis=0)
                            ligand_coords -= centroid
                            data_DL = np.concatenate((ligand_coords,ligand_features),axis=1)
                        else:
                            centroid = pocket_coords.mean(axis=0) #We use Pocket as the centroid, since with frames you can have ligand out of the binding site
                            ligand_coords -= centroid
                            if data_type == 'complex':
                                pocket_coords -= centroid
                                data_DL = np.concatenate((np.concatenate((ligand_coords, pocket_coords)),
                                                np.concatenate((ligand_features, pocket_features))), axis=1)
                            elif data_type == 'ligand':
                                data_DL = np.concatenate((ligand_coords,ligand_features),axis=1)

                        # if MD_hdf5: #not the main way for handling spatio-temporal data
                        #     LMD_data_DL.append(data_DL)
                        # else:
                        if row['include']:
                            dataset = f.create_dataset(f'{pdb}_{replicate_number}_{name}', data=data_DL, shape=data_DL.shape, dtype='float32', compression='lzf')
                            dataset.attrs['affinity'] = affinity
                            counter += 1
                        else:
                            dataset = g.create_dataset(f'{pdb}_{replicate_number}_{name}', data=data_DL, shape=data_DL.shape, dtype='float32', compression='lzf')
                            dataset.attrs['affinity'] = affinity
                            core2013_counter += 1
                    # if MD_hdf5: #not the main way for handling spatio-temporal data
                    #     #to make each frame of the same dimension
                    #     length = max(map(len,LMD_data_DL))
                    #     MD_data_DL = np.array([np.concatenate((np.array(xi,dtype='float32'),np.repeat(np.array([None]*22,dtype='float32')[np.newaxis,:],(length-len(xi)),axis=0)),axis=0) for xi in LMD_data_DL],dtype='float32')

                    #     if MD_data_DL.shape != (frame_number_total,):
                    #         continue
                        
                    #     if row['include']:
                    #         dataset = f.create_dataset(f'{pdb}_{replicate_number}', data=MD_data_DL)
                    #         dataset.attrs['affinity'] = affinity
                    #         counter += 1
                    #     else:
                    #         dataset = g.create_dataset(f'{pdb}_{replicate_number}', data=MD_data_DL, shape=MD_data_DL.shape, dtype='float32', compression='lzf')
                    #         dataset.attrs['affinity'] = affinity
                    #         core2013_counter += 1
            Dsum_up[dataset_name] = counter

    print('excluded', core2013_counter, 'complexes')
    for dataset, complexes in Dsum_up.items():
        print(f'{dataset}: prepared {complexes} complexes')
        
with h5py.File(f'{path_data_dl}/core.hdf', 'r') as f, \
     h5py.File(f'{path_data_dl}/core2013.hdf', 'r+') as g:
    for name in f:
        if name in list(split_data[split_data['core2013']].pdbid):
            dataset = g.create_dataset(name, data=f[name])
            dataset.attrs['affinity'] = f[name].attrs['affinity']
