# %load_ext autoreload
# %autoreload 2

from functools import partial
from concurrent.futures import ProcessPoolExecutor

from tqdm.auto import tqdm

from fd.sampling import sample_patch, SamplingConfig, prepare_eopatches_paths
from fd.utils import multiprocess


positive_examples_config = SamplingConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    eopatches_location='input-data/eopatches',
    output_path='input-data/patchlets',
    sample_positive=True,
    mask_feature_name='EXTENT',
    buffer=50,
    patch_size=256,
    num_samples=10,
    max_retries=10,
    fraction_valid=0.4,
    sampled_feature_name='BANDS',
    cloud_coverage=0.05)

# negatives_examples_config = SamplingConfig(
#     bucket_name='bucket-name',
#     aws_access_key_id='',
#     aws_secret_access_key='',
#     aws_region='eu-central-1',
#     eopatches_location='input-data/eopatches/with-overlap/',
#     output_path='input-data/patchlets-neg',
#     sample_positive=False,
#     grid_definition_file='input-data/cyl-grid-definition.gpkg',
#     area_geometry_file='input-data/cyl-province-border.geojson',
#     fraction_valid=0.1,
#     mask_feature_name='EXTENT',
#     patch_size=256,
#     num_samples=10,
#     max_retries=10,
#     sampled_feature_name='BANDS',
#     cloud_coverage=0.05)

eopatches_paths = prepare_eopatches_paths(positive_examples_config)
#eopatches_paths = ['input-data/eopatches/32VNH']

process_fn = partial(sample_patch, sampling_config=positive_examples_config)

_ = multiprocess(process_fun=process_fn, arguments=eopatches_paths, max_workers=24)


