from pathlib import Path
import os

import pandas as pd
import geopandas as gpd
from tqdm.notebook import tqdm

from eolearn.core import FeatureType

from fd.utils import prepare_filesystem
from fd.prediction import PredictionConfig, run_prediction_on_eopatch
from fd.prediction import load_model, load_metadata

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import json


def process_eopatches(fn, eopatches, **kwargs):
    results = []
    for eopatches_path in tqdm(eopatches):
        results.append(fn(eopatches_path, **kwargs))
    return results


def multiprocess_eopatches(fn, eopatches, max_workers, **kwargs):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        partial_fn = partial(fn, **kwargs)
        return list(tqdm(executor.map(partial_fn, eopatches), total=len(eopatches)))


def prefect_processing():
    # The idea why processing is not inside the module is to decouple it so any kind of processing can be used,
    # Either prefect, or single processing or multiprocessing or whatever
    pass


# [1] Workflow configuration set-up
INPUT_DATA_DIR = Path('input-data/')

model_version = 'folds_avg_10e'

prediction_config = PredictionConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    eopatches_folder=os.path.realpath('input-data/eopatches'),
    feature_extent=(FeatureType.DATA, f'EXTENT_PREDICTED_{model_version}'),
    feature_boundary=(FeatureType.DATA, f'BOUNDARY_PREDICTED_{model_version}'),
    feature_distance=(FeatureType.DATA, f'DISTANCE_PREDICTED_{model_version}'),
    model_path=os.path.realpath('input-data/niva-cyl-models'),
    model_name='resunet-a_avg_2022-02-10-10-20-29',  #this is difft every time we run 8trainFromCache.py
    model_version=model_version,
    temp_model_path='',  # JRCC - we already have model held locally so not used
    normalise='to_medianstd',
    height=1122,
    width=1122,
    n_channels=4,
    n_classes=2,
    metadata_path=os.path.realpath('input-data/patchlet-info.csv'),
    batch_size=16)


#filesystem = prepare_filesystem(prediction_config)

# [2] Check the meta-data used for normalisation

#normalisation_factors = load_metadata(filesystem, prediction_config)
normalisation_factors = load_metadata(prediction_config)
print(normalisation_factors)

grid_definition = gpd.read_file(INPUT_DATA_DIR/'cyl-grid-definition.gpkg')
print(grid_definition.head())

eopatches_list = grid_definition.name.values

# [3] Load model

#model = load_model(filesystem=filesystem, config=prediction_config)
model = load_model(prediction_config)


# [4] Run predictions sequentially on all patches

status = process_eopatches(run_prediction_on_eopatch,
                           eopatches_list,
                           config=prediction_config,
                           model=model,
                           normalisation_factors=normalisation_factors)

print('STATUS:')
print('**************************')
for s in status:
    print('name: ', s['name'])
    print('status: ', s['status'])
print('**************************')


status_df = pd.DataFrame(status)
print(status_df.head())
print(len(status_df), len(status_df[status_df.status=='Success']))

print('Success:')
print(status_df[status_df.status!='Success'])
print('**************************')

# [5] Check if files have been written

print('HAVE FILES BEEN WRITTEN:')
pred_files = [f'BOUNDARY_PREDICTED_{model_version}.npy',
              f'DISTANCE_PREDICTED_{model_version}.npy',
              f'EXTENT_PREDICTED_{model_version}.npy']

for eopatch in tqdm(eopatches_list):
    try:
        #files = filesystem.listdir(f'{prediction_config.eopatches_folder}/{eopatch}/data/')
        files = os.listdir(f'{prediction_config.eopatches_folder}/{eopatch}/data/')
        if not all([pf in files for pf in pred_files]):
            print(eopatch)
    except Exception as exc:
        print(f'{eopatch}: {exc}')