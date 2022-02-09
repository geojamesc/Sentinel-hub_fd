from functools import partial
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tqdm.auto import tqdm

from fd.compute_normalization import (ComputeNormalizationConfig, 
                                      stats_per_npz_ts, 
                                      prepare_filesystem,
                                      concat_npz_results,
                                      create_per_band_norm_dataframe)
from fd.utils import multiprocess

config = ComputeNormalizationConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    #npz_files_folder='input-data/patchlets_npz',
    npz_files_folder='input-data/patchletsJRCC_npz',
    #metadata_file='input-data/patchlet-info.csv'
    metadata_file='input-data/patchlet-infoJRCC.csv')

#filesystem = prepare_filesystem(config)

npz_files = os.listdir(config.npz_files_folder)
len(npz_files)



partial_fn = partial(stats_per_npz_ts, config=config)
results = multiprocess(partial_fn, npz_files, max_workers=24)

# choose here which stats you are interested in from
# ['mean', 'std', 'median', 'minimum', 'maximum', 'perc_1', 'perc_5', 'perc_95', 'perc_99']
stats_keys = ['mean', 'std', 'median', 'perc_99']
identifier_keys = ['timestamp', 'patchlet'] 

concatenated_stats = {}

for key in stats_keys+identifier_keys: 
    concatenated_stats[key] = concat_npz_results(key, results)



df = create_per_band_norm_dataframe(concatenated_stats, stats_keys, identifier_keys)

df.head()

len(df)

df.columns

# convert to datetime
timestamps = df['timestamp'].apply(lambda d: d.tz_localize(None))
df['timestamp']=timestamps.astype(np.datetime64)

# add "month" period
df['month']=df.timestamp.dt.to_period("M")

df[['mean_b0','mean_b1','mean_b2','mean_b3']].plot.box(figsize=(12,7))
plt.grid()



def plot_distributions(dataframe, stat, stat_title=None):
    colors = ['b','g','r','y']
    bands = list(range(4))
    
    if not stat_title:
        stat_title = stat

    log=True
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,13))
    for band in bands:
        dataframe.hist(f'{stat}_b{band}', ax=ax[0], range=(0,10000),
                       bins=100, log=log, color=colors[band], 
                       alpha=0.3, label=f'b{band}')
    ax[0].legend()
    ax[0].grid(axis='x')
    ax[0].set_title(f'Histograms of {stat_title}');

    log=False
    for band in bands:
        dataframe.hist(f'{stat}_b{band}', cumulative=True,  range=(0,10000),
                       density=True, ax=ax[1], bins=100, log=log, 
                       color=colors[band], alpha=0.3, label=f'b{band}')
    ax[1].legend()
    ax[1].grid(axis='x')
    ax[1].set_title(f'Cumulative distributions of {stat_title}');

plot_distributions(df, 'mean','means')



plot_distributions(df, 'perc_99','99th percentile')

aggs = {}
stat_cols = []
stats = ['perc_99', 'mean', 'median', 'std']
bands = list(range(4))
for stat in stats:
    for band in bands:
        aggs[f'{stat}_b{band}'] = [np.std, np.mean];
        stat_cols.append(f'{stat}_b{band}')

monthly = pd.DataFrame(df.groupby('month', as_index=False)[stat_cols].agg(aggs))
monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
monthly.rename(columns={'month_':'month'}, inplace=True)

monthly

def monthly_stats(monthly_df, stat, stat_title=None):
    fig, ax = plt.subplots(figsize=(12,9))
    cols = ['b','g','r','y']
    
    if not stat_title:
        stat_title = stat
        
    for band in range(4):
        x_vals = np.array([m.month for m in monthly_df['month']])
        ax.plot(x_vals, monthly_df[f'{stat}_b{band}_mean'].values, 
                color=cols[band], label=f'band {band}')
        ax.fill_between(x_vals, 
                        monthly_df[f'{stat}_b{band}_mean'].values - 
                        monthly_df[f'{stat}_b{band}_std'].values, 
                        monthly_df[f'{stat}_b{band}_mean'].values + 
                        monthly_df[f'{stat}_b{band}_std'].values, color=cols[band], 
                        alpha=0.2)

    ax.legend()
    ax.grid()
    ax.set_title(f'{stat_title} through months')

monthly_stats(monthly, 'perc_99', '99th percentiles')

norm_cols = [norm.format(band) 
             for norm in ['perc_99_b{0}_mean', 
                          'mean_b{0}_mean', 
                          'median_b{0}_mean', 
                          'std_b{0}_mean'] for band in range(4)]

def norms(month):
    return monthly.loc[monthly.month==month][norm_cols].values[0]

df['norm_perc99_b0'], df['norm_perc99_b1'], df['norm_perc99_b2'], df['norm_perc99_b3'], \
df['norm_meanstd_mean_b0'], df['norm_meanstd_mean_b1'], df['norm_meanstd_mean_b2'], df['norm_meanstd_mean_b3'], \
df['norm_meanstd_median_b0'], df['norm_meanstd_median_b1'], df['norm_meanstd_median_b2'], df['norm_meanstd_median_b3'], \
df['norm_meanstd_std_b0'], df['norm_meanstd_std_b1'], df['norm_meanstd_std_b2'], df['norm_meanstd_std_b3'] = zip(*map(norms, df.month))

df[['month','norm_perc99_b0','norm_perc99_b1','norm_perc99_b2','norm_perc99_b3']].drop_duplicates()

# this plot should reflect solid lines from `monthly_stats(monthly, 'perc_99','99th percentiles')`
df[['month','norm_perc99_b0','norm_perc99_b1','norm_perc99_b2','norm_perc99_b3']].drop_duplicates().reset_index(drop=True).plot()

# another check; should be similar to `monthly_stats(monthly, 'mean','means')`
df[['month','norm_meanstd_mean_b0','norm_meanstd_mean_b1','norm_meanstd_mean_b2','norm_meanstd_mean_b3']].drop_duplicates().reset_index(drop=True).plot()

with open(config.metadata_file, 'rb') as fcsv:
    df_info = pd.read_csv(fcsv)

df_info['timestamp'] = pd.to_datetime(df_info.timestamp)

timestamps = df_info['timestamp'].apply(lambda d: d.tz_localize(None))
df_info['timestamp'] = timestamps.astype(np.datetime64)

df_info.head()

new_df = df_info.merge(df, how='inner', on=['patchlet', 'timestamp'])



with open(config.metadata_file, 'w') as fcsv:
    new_df.to_csv(fcsv, index=False)

