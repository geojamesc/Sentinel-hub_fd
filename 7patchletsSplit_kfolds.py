import os 
from functools import partial

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt 

from fd.utils import multiprocess, prepare_filesystem
from fd.training import SplitConfig, fold_split

split_config = SplitConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    metadata_path='input-data/patchlet-info.csv',
    npz_folder='input-data/patchlets_npz',
    n_folds=3
) #Too small area for more?

#filesystem = prepare_filesystem(split_config)

df = pd.read_csv(split_config.metadata_path)

df.head()

eops = df.eopatch.unique()
print('eops:', eops)

np.random.seed(seed=split_config.seed)

print('len(eops): ', len(eops))
print('split_config.n_folds: ', split_config.n_folds)

# this seems to often generate a set of values with duplicates
#fold = np.random.randint(1, high=split_config.n_folds+1, size=len(eops))
fold = np.array([3, 1, 2])

print('fold: ', fold)
eopatch_to_fold_map = dict(zip(eops, fold))

print(eopatch_to_fold_map)

df['fold'] = df['eopatch'].apply(lambda x: eopatch_to_fold_map[x])

print(df)

for nf in range(split_config.n_folds):
    print(f'{len(df[df.fold==nf+1])} patchlets in fold {nf+1}')

gdf = gpd.read_file('input-data/cyl-grid-definition.gpkg')

gdf.head()

len(eops), len(gdf)

gdf_training = gdf[gdf['name'].isin(eops)]

gdf_training['fold'] = gdf_training.name.apply(lambda x: eopatch_to_fold_map[x])

fig, ax = plt.subplots(figsize=(15, 15))
gdf_training.plot(ax=ax, column='fold')
gdf.boundary.plot(ax=ax)

partial_fn = partial(fold_split, df=df, config=split_config)

npz_files = os.listdir(split_config.npz_folder)

npz_files = [npzf for npzf in npz_files if npzf.startswith('patchlets_')]

len(npz_files)

_ = multiprocess(partial_fn, npz_files, max_workers=20)


with open(split_config.metadata_path, 'w') as fcsv:
    df.to_csv(fcsv, index=False)

for fold in range(1, split_config.n_folds+1):
    
    fold_folder = os.path.join(split_config.npz_folder, f'fold_{fold}')

    for i in [0]: #7, 50, 154, 176, 185
        chunk = f'patchlets_field_delineation_{i}.npz'

        data = np.load(os.path.join(fold_folder, chunk))

        fold_df = df[(df.chunk == chunk) & (df.fold == fold)]
        
        assert len(fold_df.chunk_pos) == data['y_boundary'].shape[0]
        assert len(fold_df.chunk_pos) == data['y_extent'].shape[0]
        assert len(fold_df.chunk_pos) == data['y_distance'].shape[0]
        assert len(fold_df.chunk_pos) == data['X'].shape[0]
        
        print(f"For fold_{fold}/{chunk} the lengths match.")

