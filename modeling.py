#!/usr/bin/env python
# coding: utf-8

#%%
# import libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.weightstats import ztest
from statsmodels.formula.api import ols
from statsmodels.stats.anova import *
from scipy.stats import ttest_1samp
from scipy.stats import chi2_contingency, chisquare
# import contextily as ctx

# import contextily as ctx
#%% 


# Data sources

# | Variable Name | Variable URL |
# | :-: | :-: |
# | Income census data | https://api.census.gov/data/2017/acs/acs1/groups/B25074.html |
# | Census tract  | https://www2.census.gov/geo/tiger/TIGER2017/TRACT/tl_2017_11_tract.zip |
# | MetroBus stops | https://opendata.arcgis.com/datasets/e85b5321a5a84ff9af56fd614dab81b3_53.geojson |
# | Metro Station | https://opendata.arcgis.com/datasets/ab5661e1a4d74a338ee51cd9533ac787_50.geojson |
# | DC buildings | https://opendata.arcgis.com/datasets/8ffa9109cd9a4e37982cea67b289784d_0.geojson |

#%% [markdown]
# ## Census variables 

# link to census ID and description
# https://api.census.gov/data/2019/acs/acs1/groups.html

# | Variable Name | Variable Description |
# | :-: | :-: |
# | B02001_002E | Population of White race |
# | B02001_003E | Population of Black/ Africa American Race |
# | B25074_020E | Household income by gross rent as a percentage of household in the past 12 months |
# | B01003_001E | Total population |

# census variables

#%%
# Population :
# White population :
# Black population :
# Household income :

# selected infrastructure 

# Bus stops :
# Public schools :
# Metro station :
# Trees :
# Parks :


#%%
# read geojosn data with geopandas

df = gpd.read_file('./data/dataset.geojson')
print(df.head())
df.fillna(0)
# %%

df.plot()


#%%

















