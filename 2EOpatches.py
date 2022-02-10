from pathlib import Path

from typing import Callable, List, Any

import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from eolearn.core import EOPatch, EOExecutor

from fd.tiffs_to_eopatch import (
    TiffsToEopatchConfig,
    get_tiffs_to_eopatches_workflow,
    get_exec_args)

from fd.utils import prepare_filesystem



INPUT_DATA_DIR = Path('input-data/')


tiffs_to_eops_config = TiffsToEopatchConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    tiffs_folder='input-data/tiffs',
    eopatches_folder='input-data/eopatches',
    band_names=['B02', 'B03', 'B04', 'B08'],
    mask_name='dataMask',
    clp_name='CLP',
    clm_name='CLM'
)

workflow = get_tiffs_to_eopatches_workflow(tiffs_to_eops_config, delete_tiffs=False)

grid_definition = gpd.read_file(INPUT_DATA_DIR/'cyl-grid-definition.gpkg')
#grid_definition.to_file("input-data/cyl-province-border.geojson",  driver="GeoJSON")

eopatch_list = grid_definition.name.values

exec_args = get_exec_args(workflow, eopatch_list)

exec_args[0]


eop = workflow.execute(exec_args[0])
eop = list(eop.values())[0]

print(eop)
print(eop.timestamp)


tidx = 0

#plt.figure(figsize=(15,15))
#plt.imshow(2.5*eop.data['BANDS'][tidx][..., [2,1,0]]/10000)
#plt.imshow(eop.mask['CLM'][tidx][..., 0], vmin=0, vmax=1, alpha=.2)

MAX_WORKERS = 24

executor = EOExecutor(workflow, exec_args, save_logs=True, logs_folder='.')

executor.run(workers=MAX_WORKERS)

#%matplotlib

executor.make_report()

print('Report was saved to location: {}'.format(executor.get_report_filename()))



