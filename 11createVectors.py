import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import glob
import logging
import sys
import pprint

from fd.vectorisation import (
    VectorisationConfig,
    get_weights, 
    write_vrt, 
    spatial_merge_contours,
    run_vectorisation
)


# this code will produce a (possibly extremely large) GeoTIFF
# use with caution!
def vrt2tiff(vrt, tiff):
    with rasterio.Env(GDAL_VRT_ENABLE_PYTHON=True):
        with rasterio.open(vrt, 'r') as src:
            meta = src.meta
            data = src.read(1)
            meta.update(driver='gTiff')

            with rasterio.open(tiff,'w', **meta) as dst:
                dst.write(data, 1)


vectorisation_config = VectorisationConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    tiffs_folder=os.path.realpath('results/Denmark'),
    time_intervals=['FEB'],
    utms=['32632'],
    shape=(6032, 6032),
    buffer=(200, 200),
    weights_file=os.path.realpath('cyl-weights.tiff'),
    vrt_dir=os.path.realpath('vrt'),
    predictions_dir=os.path.realpath('cyl-predictions'),
    contours_dir=os.path.realpath('cyl-contours'),
    max_workers=20
)

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)



# JRCC - before running this make sure the FEB folder and it`s contents are copied from
# results/Denmark to cyl-predictions. These files like 32VNH_1-32632.tiff are what was
# produced by 10postProcessPredictions.py

list_of_merged_files = run_vectorisation(vectorisation_config)
pprint.pprint(list_of_merged_files)

