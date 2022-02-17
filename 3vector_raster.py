from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from sentinelhub import CRS, BBox

from eolearn.core import FeatureType, EOExecutor

from fd.gsaa_to_eopatch import GsaaToEopatchConfig, get_gsaa_to_eopatch_workflow

import warnings
from fd.utils_plot import (draw_vector_timeless, 
                           draw_true_color, 
                           draw_bbox, 
                           draw_mask, 
                           get_extent
                          )

INPUT_DATA_DIR = Path('input-data/')

gsaa_to_eops_config = GsaaToEopatchConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    database='gisdb',
    user='niva',
    password='n1v4',
    host='localhost',
    port='25431',
    crs=CRS.WGS84.pyproj_crs(),
    eopatches_folder='input-data/eopatches',
    vector_feature=(FeatureType.VECTOR_TIMELESS, 'GSAA_ORIGINAL'),
    extent_feature=(FeatureType.MASK_TIMELESS, 'EXTENT'),
    boundary_feature=(FeatureType.MASK_TIMELESS, 'BOUNDARY'),
    distance_feature=(FeatureType.DATA_TIMELESS, 'DISTANCE'),
    height=1200,
    width=1200
)

grid_definition = gpd.read_file(INPUT_DATA_DIR/'cyl-grid-definition.gpkg')

eopatches_list = grid_definition.name.values
#eopatches_list[0]
print('eopatches_list: ', eopatches_list)


workflow = get_gsaa_to_eopatch_workflow(gsaa_to_eops_config)

tasks = workflow.get_tasks()

#result = workflow.execute({
#    tasks['LoadTask']: {'eopatch_folder': '32VNH_1'},
#    tasks['SaveTask']: {'eopatch_folder': '32VNH_1'}
#    })
#eop = list(result.values())[-1]

exec_args = []

for eopatch_name in tqdm(eopatches_list, total=len(eopatches_list)):
    single_exec_dict = {}
    single_exec_dict[tasks['LoadTask']] = dict(eopatch_folder=f'{eopatch_name}')
    single_exec_dict[tasks['SaveTask']] = dict(eopatch_folder=f'{eopatch_name}')
    exec_args.append(single_exec_dict)

warnings.filterwarnings("ignore", category=DeprecationWarning) 

MAX_WORKERS = 24

executor = EOExecutor(workflow, exec_args, save_logs=True, logs_folder='.')

executor.run(workers=MAX_WORKERS)


executor.make_report()

print('Report was saved to location: {}'.format(executor.get_report_filename()))

#failed = !grep failed -l /home/ubuntu/field-delineation/notebooks/data-download/eoexecution-report-2020_12_28-21_58_36/*log | awk '{system("grep eopatch_folder "$0)}' | awk -F "'" '{print $4}'

tiles = grid_definition[['name','geometry']].copy()

len(tiles)

tiles['failed']=False


#tiles.loc[tiles['name'].isin(failed), ['failed']]=True

