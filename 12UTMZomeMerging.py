#!/usr/bin/env python
# coding: utf-8

# ## Merge two UTM zones
# 
# ```
# #
# # Copyright (c) Sinergise, 2019 -- 2021.
# #
# # This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# # All rights reserved.
# #
# # This source code is licensed under the MIT license found in the LICENSE
# # file in the root directory of this source tree.
# #
# ```

# This notebook implements the methods needed to merge two UTM zones into a single output geopackage.
# 
# The procedure outline is:
# * define geometries for two UTM zones and their overlap
# * load the two single-UTM-zone vector predictions
# * split them into parts: non-overlapping (completely within UTM zone) and overlapping
# * merge the overlaps by:
#   * transform them to single CRS (WGS84)
#   * spatial join of the overlapping geodataframes from the two zones
#   * finding geometries that do not overlap (and keeping them)
#   * unary_union-ize the polygons that intersect and merge them to the geometries from previous step
# * transform everything to resulting (common) CRS
# * clean up the results (remove geometries with area larger than X * largest GSAA polygon from 2020
# * simplify geometries

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union

from fd.vectorisation import MergeUTMsConfig, utm_zone_merging


# ### The two UTM zones and their overlap

# In[3]:


INPUT_DATA_DIR = Path('../../input-data/')

grid_definition = gpd.read_file(INPUT_DATA_DIR/'cyl-grid-definition.gpkg')
grid_definition.head()


# In[4]:


grid_definition['crs'] = grid_definition['name'].apply(lambda name: f'326{name[:2]}')
grid_definition.head()


# In[5]:


crs_names = sorted(grid_definition['crs'].unique())
crs_names


# In[6]:


utm_geoms = [grid_definition[grid_definition['crs']==crs_name].geometry.unary_union 
             for crs_name in crs_names]


# In[7]:


overlap = utm_geoms[0].intersection(utm_geoms[1]).buffer(-0.0001)


# In[8]:


tiled_overlap = grid_definition[grid_definition.intersects(overlap)].unary_union.buffer(-0.0001)


# In[9]:


zones = gpd.GeoDataFrame(geometry=[g for g in grid_definition[~grid_definition.intersects(tiled_overlap)].buffer(0.00001).unary_union],
                         crs=grid_definition.crs)
zones['crs'] = zones.geometry.apply(lambda g: grid_definition[grid_definition.intersects(g)]['crs'].unique()[0])


# In[10]:


# a dataframe holding the overlap geometry
# useful because it is much simpler to transform between CRSs 

overlap_df = gpd.GeoDataFrame(geometry=[tiled_overlap], crs=grid_definition.crs)


# In[11]:


fig, ax = plt.subplots(figsize=(15,10))
zones.plot('crs', ax=ax)
overlap_df.boundary.plot(ax=ax, color='r')


# In[12]:


MAX_GSAA_AREA = 19824325 # m2, derived from reference data

merging_config = MergeUTMsConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    time_intervals=['APRIL'],
    utms=crs_names,
    contours_dir='/home/ubuntu/cyl-contours', # where partial vectors are stored
    resulting_crs='epsg:2062', # CRS of resulting geometries
    max_area=1.3*MAX_GSAA_AREA, # Specify max area in m2 of parcels to keep
    simplify_tolerance=2.5, # This is in teh resulting CRS, careful about unit of measure
    n_workers=20
)


# In[13]:


import logging
import sys

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)


# Depending on the number of polygons in the overlap area, and on their size, this step can take quite a long time (appr 4 hours per time interval).

# In[14]:


utm_zone_merging(merging_config, overlap_df, zones, parallel=False)

