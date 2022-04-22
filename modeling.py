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
plt.figure()
df.plot()
plt.show()

#%%
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(15,15))
axes[0,0].set_title("Bus stops counts per census tract") 
df.plot(ax=axes[0,0], column='bus_stops', scheme='NaturalBreaks', k=5, \
             cmap='YlOrRd', legend=True,
             legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
          )
  
axes[0,1].set_title("Tree counts per census tract")
df.plot(ax=axes[0,1], column='tree', scheme='NaturalBreaks', k=5, \
             cmap='YlOrRd', legend=True,
             legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
          )

axes[1,0].set_title("Public schools counts per census tract") 
df.plot(ax=axes[1,0], column='public_school', scheme='NaturalBreaks', k=5, \
             cmap='YlOrRd', legend=True,
             legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
          )

axes[1,1].set_title("Metro station counts per census tract") 
df.plot(ax=axes[1,1], column='metro_station', scheme='NaturalBreaks', k=5, \
             cmap='YlOrRd', legend=True,
             legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
          )
# %%
fig, axes = plt.subplots(figsize=(10, 10))  
axes.set_title("Bus stops counts per census tract") 
df.plot(ax=axes, column='bus_stops', scheme='NaturalBreaks', k=5, \
             cmap='YlOrRd', legend=True,
        #      legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
             missing_kwds={"color": "lightgrey", "edgecolor": "red", "hatch": "///","label": "Missing values"}
          )
plt.show()

# %%
fig, axes = plt.subplots(figsize=(10, 10))  
axes.set_title("Tree counts per census tract")
df.plot(ax=axes, column='tree', scheme='NaturalBreaks', k=5, \
             cmap='YlOrRd', legend=True,
             legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True, "color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "No metro station"},
          )
plt.show()
# df.plot(
#     column="metro_station",
#     legend=True,
#     scheme="quantiles",
#     figsize=(15, 10),
#     missing_kwds={
#         "color": "lightgrey",
#         "edgecolor": "red",
#         "hatch": "///",
#         "label": "No metro station",
#     },
# )
# %%
# This transformation does not make sense to me.

df.loc[df['White_population'] > df['Black_population'], 'Race'] = 'White'
df.loc[df['White_population'] < df['Black_population'], 'Race'] = 'Black_Africa'
print(df.head())

# %%
fig, axes = plt.subplots(figsize=(10, 10))
axes.set_title("White and Black/Africa American in DC")
df.plot(ax=axes, column='Race', \
             cmap=plt.cm.get_cmap('Dark2', 2).reversed(), legend=True,
             )
plt.show()
# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
df.plot(ax=axes[0], column='Race', \
        cmap=plt.cm.get_cmap('Dark2', 2).reversed(), legend=True)
axes[0].set_title("Public School distributions")

df.plot(ax=axes[1], column='tree', scheme='NaturalBreaks', k=5, \
        cmap='YlOrRd', legend=True,
        legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5), 'interval': True})
axes[1].set_title(" per census tract")
plt.show()

# You are erasing Race attribute by reloading data here.
#df = gpd.read_file('data/dataset.geojson')
#print(df.head()) # No need to repeat this

#%%[markdown]
# Chapter: EDA


#%%[markdown]
# How race is distributed in data
total_black_afr = np.sum(df.Race=='Black_Africa')
total_white_amr = np.sum(df.Race=='White')
print(f'There are in total {total_black_afr} regions mostly occupied by Black Africans, while {total_white_amr} regions'
      f' mostly occupied by White Americans.')

#%%[markdown]
# Visualizing what relationship does highly accumulated region of particular race has with the household income
bp_hi = ols('Household_income~Black_population', data=df).fit()
wp_hi = ols('Household_income~White_population', data=df).fit()
tmp_params1 = bp_hi.params
tmp_params2 = wp_hi.params

bp_range = np.linspace(df.Black_population.min(), df.Black_population.max(), 100)
wp_range = np.linspace(df.White_population.min(), df.White_population.max(), 100)
bp_hi = bp_range * tmp_params1.Black_population + tmp_params1.Intercept
wp_hi = wp_range * tmp_params2.White_population + tmp_params2.Intercept
plt.figure()
plt.plot(df.Black_population, df.Household_income, '.b', label='Black population')
plt.plot(bp_range, bp_hi, '-b')
plt.plot(df.White_population, df.Household_income, '.r', label='White population')
plt.plot(wp_range, wp_hi, '-r')
plt.xlabel('Population')
plt.ylabel('Household Income')
plt.title('Relationship of household income with population of a particular race ')
plt.legend()
plt.show()

#%%[markdown]
# Visualization of Black African and White American population
plt.figure()
plt.hist(df.loc[:, ['Black_population', 'White_population']], stacked=False, bins=10,
         label=['Black Population', 'White Population'])
plt.xlabel('Black and White Population')
plt.ylabel('Frequency count')
plt.title('Black and White Population distribution')
plt.legend()
plt.show()


#%%[markdown]
# Visualization of how Household Income is distributed
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(data=df.loc[:, ['Household_income', 'Race']], x='Household_income', hue='Race',
             hue_order=['Black_Africa', 'White'], legend=True, ax=axes[0])
sns.boxplot(x='Race', y='Household_income', data=df, ax=axes[1])
plt.xlabel('Household Income')
plt.ylabel('Frequency count')
plt.legend(['Black_Africa', 'White'])
plt.show()

#%%[markdown]

_, race_hhi_2z_pval = ztest(df.loc[df.Race=='Black_Africa', 'Household_income'],
                            df.loc[df.Race=='White', 'Household_income'],
                            alternative='two-sided', value=0)
# I ran a z.test to see if the difference in household income between the two subgroups Black_Africa
# and White is statistically significant and t.test produced a p.value less than 0.05 confirming
# difference is statistically significant.
# P.S: Z test was chosen since, although household income distribution is right skewed, we have enough observations
# required to run z test and given that data is truly the entire population so that sd here means the true population
# statistic
print(f'p.value = {race_hhi_2z_pval}')

# t.test on whether Black Africans earn Household income of at least 10% more than what White Americans make in the DC
threshold = df.loc[df.Race=='White', 'Household_income'].mean() * 1.1
_, race_hhi_1t_pval = ttest_1samp(df.loc[df.Race=='Black_Africa', 'Household_income'],
                                  popmean=threshold, alternative='less')
print(f'p.value = {race_hhi_1t_pval}')

#%%[markdown]

df['wtob'] = df.White_population/df.Black_population
fig, axes = plt.subplots(figsize=(10, 10))
axes.set_title("White and Black/Africa American in DC")
# df.plot(ax=axes, column='wtob', cmap=plt.cm.get_cmap('Dark2', 2).reversed(), legend=True)
df.explore(column='Black_population')
plt.show()

















