import os
import warnings
from pathlib import Path
import geopandas as gpd
from eolearn.core import FeatureType, EOExecutor
from fd.post_processing import (
    get_post_processing_workflow,
    get_exec_args,
    PostProcessConfig)
import pprint

warnings.simplefilter(action='ignore', category=UserWarning)


INPUT_DATA_DIR = Path('input-data/')
model_version = 'folds_avg_10e'


# time_intervals is the range of dates of data from a start week number to an end week number
# using the iso calander (as opposed to gregorian normal calander)
# synergise assume we are dealing with data for the same year, so in post_processing.py they only check week number, not year
post_process_config = PostProcessConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    #time_intervals=dict(APRIL=(12, 21)), # include last weeks of March and firsts of May [JRCC they mean week number 12 --> 21 in some year]
    time_intervals=dict(FEB=(8, 24)),  # JRCC our data is for different years but spans week numbers 9 (20190227) --> 23 (20200601)
    eopatches_folder=os.path.realpath('input-data/eopatches'),
    tiffs_folder=os.path.realpath('results/Denmark'),
    feature_extent=(FeatureType.DATA, f'EXTENT_PREDICTED_{model_version}'),
    feature_boundary=(FeatureType.DATA, f'BOUNDARY_PREDICTED_{model_version}'),
    model_version=model_version,
    #max_cloud_coverage=0.10 # increased from 0.05 since there aren't good frames available
    max_cloud_coverage=0.25  # JRCC in our data max_cloud_coverage is always like 0.21121789008576716
)


# #### List of patches
grid_definition = gpd.read_file(INPUT_DATA_DIR/'cyl-grid-definition.gpkg')
print('grid_definition:')
print(grid_definition.head())

eopatches_list = grid_definition.name.values
print('eopatches_list: ')
print(eopatches_list)

# ### Get workflow

workflow = get_post_processing_workflow(post_process_config)
exec_args = get_exec_args(workflow=workflow,
                          eopatch_list=eopatches_list,
                          config=post_process_config)

#print('exec_args:')
#for arg in exec_args:
#    pprint.pprint(arg)


# Multi-process using `EOExecutor`

executor = EOExecutor(workflow, exec_args, save_logs=True)
executor.run(workers=20)
executor.make_report()