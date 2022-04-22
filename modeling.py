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

#%% [markdown]
# Is there segregation in public infrastructure investment in the DC?

# plot Bus stops 
fig, axes = plt.subplots(figsize=(10, 10), sharex=True)  
axes.set_title('Bus stops counts per census tracts') 
df.plot(ax=axes, column= 'bus_stops', scheme='NaturalBreaks', k=5, \
            cmap='YlOrRd', legend=True,
            legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
         )
plt.show()

#%%
# Trees
fig, axes = plt.subplots(figsize=(10, 10), sharex=True)  
axes.set_title('Tree counts per census tract') 
df.plot(ax=axes, column= 'tree', scheme='NaturalBreaks', k=5, \
            cmap='YlOrRd', legend=True,
            legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
         )
plt.show()

#%%

# Plot of aggregated infrastructure per census tract

def plot_infrastructure(variable, title, missing_data):
   '''A function to plot infrastrucutre
   param: variable and title in string
   return a map'''
   
   fig, axes = plt.subplots(figsize=(10, 10), sharex=True)  
   axes.set_title(title) 
   missing_kwds = dict(color='grey', label=missing_data)
   df.plot(ax=axes, column= variable, scheme='NaturalBreaks', k=5, \
               cmap='YlOrRd', legend=True,
               legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
               missing_kwds = missing_kwds
            )
   plt.show()


#%%
# Public schools
plot_infrastructure('public_school', 'Public school counts per census tracts', 'No Public Schools')

#%%
# Metro stations
plot_infrastructure('metro_station','Metro stations count per census tracts', 'No Metro Station')

#%%


#%%
# Parks
plot_infrastructure('park', 'Park counts per census track', 'No Parks' )

#%%
# plot Census variables 

# Assume predominant race in the census tract 
df.loc[df['White_population'] > df['Black_population'], 'Race'] = 'White'
df.loc[df['White_population'] < df['Black_population'], 'Race'] = 'Black_Africa'
print(df.head())

#%%
fig, axes = plt.subplots(figsize=(10, 10))
axes.set_title("White and Black/Africa American in DC")
df.plot(ax=axes, column='Race', \
             cmap=plt.cm.get_cmap('Dark2', 2).reversed(), legend=True,
             )
plt.show()

# Visually there is seems to be clusters of race in DC. 

#%%
# df.fillna(0)
df1 =df.dropna()
# Logistic model 
x = df1[['Household_income', 'bus_stops', 'tree']]
y = df1['Race']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1 )

# logistric regression
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()  # instantiate
logit.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', logit.score(x_test, y_test))
print('Logit model accuracy (with the train set):', logit.score(x_train, y_train))
#%%
df['lg_pred'] = logit.predict(x)
#%%
fig, axes = plt.subplots(figsize=(10, 10))
axes.set_title("White and Black/Africa American in DC")
df.plot(ax=axes, column='lg_pred', \
             cmap=plt.cm.get_cmap('Dark2', 2).reversed(), legend=True,
             )
plt.show()
#%%

#%%
# Radnom forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, random_state = 0)

rf.fit (x_train, y_train)
print('Random forest model accuracy (with the test set):', rf.score(x_test, y_test))
print('Random forest model accuracy (with the train set):', rf.score(x_train, y_train))
#%%
df['rf_pred'] = rf.predict(x)
#%%
fig, axes = plt.subplots(figsize=(10, 10))
axes.set_title("White and Black/Africa American in DC")
df.plot(ax=axes, column='rf_pred', \
             cmap=plt.cm.get_cmap('Dark2', 2).reversed(), legend=True,
             )
plt.show()

# print(rf.predict(x_test))
#%%







# %%
