import os

import numpy as np
import pandas as pd 
from tqdm.auto import tqdm

from functools import partial 
from concurrent.futures import ProcessPoolExecutor

from fd.utils import prepare_filesystem, multiprocess
from fd.create_npz_files import (
    CreateNpzConfig, 
    extract_npys, 
    concatenate_npys, 
    save_into_chunks
)

config = CreateNpzConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    patchlets_folder='input-data/patchlets',
    output_folder='input-data/patchlets_npz',
    output_dataframe='input-data/patchlet-info.csv',
    chunk_size=50)
    
#filesystem = prepare_filesystem(config)

patchlets = [os.path.join(config.patchlets_folder, eop_name)
             for eop_name in os.listdir(config.patchlets_folder)]

print('len - patchlets: ', len(patchlets))

partial_fn = partial(extract_npys, cfg=config)

npys = multiprocess(partial_fn, patchlets, max_workers=24)

npys_dict = concatenate_npys(npys)

print('npys_dict_keys: ', npys_dict.keys())
print('len_npys_dict:', len(npys_dict))

print('npys_dict_timestamps:', npys_dict['timestamps'])
#print('npys_dict_X:', npys_dict['X'])

save_into_chunks(config, npys_dict)

npzs = os.listdir(config.output_folder)

print('len_npzs: ', len(npzs))

test_npz = np.load(os.path.join(config.output_folder, npzs[0]), 
                   allow_pickle=True)

test_npz['X'].shape, test_npz['y_extent'].shape, test_npz['timestamps'].shape

df = pd.read_csv(config.output_dataframe) #filesystem.open(config.output_dataframe))

df.head()

len(df)
