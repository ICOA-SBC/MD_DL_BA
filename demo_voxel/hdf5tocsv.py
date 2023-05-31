import argparse
import os
from pathlib import Path

import h5py
import numpy as np

"""
Build csv files from hdf5 format

Output files:

output_dir \ pdb_id1.csv 
           \ pdb_id2.csv
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Interpret hdf5 and build 3d volume dataset - expext 2 hdf5 archives (train and validation)', )
    parser.add_argument('-i', '--input_h5', required=True, default="", type=str,
                        help='complete input filename (h5 format)')
    parser.add_argument('-d', '--output_format', required=True, default="2", type=int,
                        help='output data format (number of digits :.%df)')
    parser.add_argument('-o', '--output_dir', required=True, default="", type=str,
                        help='complete output path')
    args = parser.parse_args()

    return args


def save_complex_as_csv(output_dir, pdb_id, content, output_format):
    output_file = os.path.join(output_dir, pdb_id + ".csv")
    np.savetxt(output_file, content, fmt=output_format, delimiter=' ', newline='\n')


def main(args):
    input_h5 = args.input_h5
    output_format = args.output_format if args.output_format < 15 else 6
    output_dir = args.output_dir+str(output_format)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_format = f'%.{output_format}f'

    with h5py.File(input_h5, "r") as f:
        for idx, pdb_id in enumerate(f):  
            data = f[pdb_id]  # hdf5 dataset
            content = data[:]  # numpy array
            save_complex_as_csv(output_dir, pdb_id, content, output_format)
    print(f"{idx+1} samples processed.")

            
if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)

"""
python hdf5tocsv.py -i $DATA_COG14/validation_set.hdf -d "2" -o $DATA_CSV/validation
python hdf5tocsv.py -i $DATA_COG14/validation_set.hdf -d "6" -o $DATA_CSV/validation
> 1000 samples processed.
python hdf5tocsv.py -i $DATA_COG14/training_set.hdf -d "2" -o $DATA_CSV/training
python hdf5tocsv.py -i $DATA_COG14/training_set.hdf -d "6" -o $DATA_CSV/training
> 16279 samples processed.

Folder size:

du -h $DATA_CSV/validation2
du -h $DATA_CSV/validation6
du -h $DATA_CSV/training2
du -h $DATA_CSV/training6

(python-3.10.4) du -h $DATA_CSV/validation2
82M     $SCRATCH/data/frugalpython/csv/validation2
(python-3.10.4) du -h $DATA_CSV/validation6
140M    $SCRATCH/data/frugalpython/csv/validation6
(python-3.10.4) du -h $DATA_CSV/training2
1.3G    $SCRATCH/data/frugalpython/csv/training2
(python-3.10.4) du -h $DATA_CSV/training6
2.1G    $SCRATCH/data/frugalpython/csv/training6

"""
