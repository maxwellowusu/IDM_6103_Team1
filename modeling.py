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
#%%[markdown]
## Smart Question 3
# Is there a relationship between distribution of infrastructure and socio-economic  and race groups?
#%%[markdown]
# In order to do some data process, define a new dataframe. 
# Calculate the ration of white_population to Total Population, and add it to a new column named race_ratio. 
#%%
df3 = df
df3["race_ratio"]=df3[["White_population","Population"]].apply(lambda x:x["White_population"]/x["Population"],axis=1)
df3.head()
#%%
# Build regression model between income and races: 
import statsmodels.api as sm
fit_income=sm.formula.ols('Household_income~race_ratio',data = df3).fit()
print(fit_income.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is less than 0.05, we reject the null hypothesis in favor of alternative hypothesis. The race ratio is negative ralated to household income. That as the white population increse, the household income would decrease. 

#%%
# Build regression model between bus_stops and races: 
import statsmodels.api as sm
fit_bus=sm.formula.ols('bus_stops~race_ratio',data = df3).fit()
print(fit_bus.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is higher than 0.05, we fail to reject the null hypothesis. The race ratio would not have statistical significant effect on bus stops. 
#%%
# Build regression model between public_school and races: 
import statsmodels.api as sm
fit_school=sm.formula.ols('public_school~race_ratio',data = df3).fit()
print(fit_school.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is higher than 0.05, we fail to reject the null hypothesis. The race ratio would not have statistical significant effect on the number of public schools. 
#%%
# Build regression model between metro_station and races: 
import statsmodels.api as sm
fit_metro=sm.formula.ols('metro_station~race_ratio',data = df3).fit()
print(fit_metro.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is less than 0.05, we reject the null hypothesis in favor of alternative hypothesis. The race ratio is positive ralated to metro station. That as the white population increse, the metro station would also increse. 
#%%
# Build regression model between tree and races: 
import statsmodels.api as sm
fit_tree=sm.formula.ols('tree~race_ratio',data = df3).fit()
print(fit_tree.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is higher than 0.05, we fail to reject the null hypothesis. The race ratio would not have statistical significant effect on the number of trees. 

#%%
# Build regression model between park and races: 
import statsmodels.api as sm
fit_park=sm.formula.ols('park~race_ratio',data = df3).fit()
print(fit_park.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is higher than 0.05, we fail to reject the null hypothesis. The race ratio would not have statistical significant effect on the number of parks. 

#%%[markdown]
## Smart Question 4
# How people of different economic status are affected by this infrastructure development?
#%%
# In order to do some data process, define a new dataframe. 
df4 = df
#%%[markdown]
# Before inferential analysis, we do some descriptive analysis. 
#%%
print(df4.info())

countsschool = df4['public_school'].value_counts()
plt.pie(countsschool, labels = countsschool.index, startangle = 90,
        counterclock = False, autopct='%.1f%%')
plt.axis('square')
plt.legend()
plt.title('Number of Public School')
plt.show()

countsmetro = df4['metro_station'].value_counts()
plt.pie(countsmetro, labels = countsmetro.index, startangle = 90,
        counterclock = False, autopct='%.1f%%')
plt.axis('square')
plt.legend()
plt.title('Number of Metro Stations')
plt.show()

countspark = df4['park'].value_counts()
plt.pie(countspark, labels = countspark.index, startangle = 90,
        counterclock = False, autopct='%.1f%%')
plt.axis('square')
plt.legend()
plt.title('Number of Parks')
plt.show()

#%%[markdown]
# From the plots above, we can see that facilities are not built equally in each region. 

#%%
ax = sns.distplot(df4['Household_income'])
ax.set_xlabel("Household Income", fontsize=20)
ax.set_ylabel("Density", fontsize=20)
plt.title('Density of Household Income')
plt.show()


kind='reg' 
ax = sns.jointplot(x='bus_stops', y='Household_income', data=df4, kind='reg', height=5)
ax.set_axis_labels('Bus Stops', 'Household Income', fontsize=20)  
plt.show()

kind='reg' 
ax = sns.jointplot(x='public_school', y='Household_income', data=df4, kind='reg', height=5)
ax.set_axis_labels('Public Schools', 'Household Income', fontsize=20)  
plt.show()

# Since the range of public school is only includes 1, 2, 3, 4, we decide to draw a boxplot which is more readable. 
ax = sns.boxplot(x='public_school', y='Household_income', data=df4)
ax.set_xlabel('Public Schools', fontsize=20)
ax.set_ylabel('Household Income', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

kind='reg' 
ax = sns.jointplot(x='metro_station', y='Household_income', data=df4, kind='reg', height=5)
ax.set_axis_labels('Metro Stations', 'Household Income', fontsize=20)  
plt.show()

kind='reg' 
ax = sns.jointplot(x='tree', y='Household_income', data=df4, kind='reg', height=5)
ax.set_axis_labels('Trees', 'Household Income', fontsize=20)  
plt.show()

kind='reg' 
ax = sns.jointplot(x='park', y='Household_income', data=df4, kind='reg', height=5)
ax.set_axis_labels('Praks', 'Household Income', fontsize=20)  
plt.show()

#%%[markdown]
# Joint distribution plots above shows that bus stop would have positive effect on household income, and metro station, tree and park would be negative related to household income. 
# 
# Then we build a model to see the coefficient. 

#%%
import statsmodels.api as sm
# %%
fit41=sm.formula.ols('Household_income~bus_stops+public_school+metro_station+tree+park',data = df4).fit()
print(fit41.summary())

# %%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
#
# Here we can see from the coefficients that metro_station, tree and park are negative related to household_income, and bus_stops and public_school are positive related to household_income. 
# 
# However, take $\alpha=0.05$, the p-values of all variables are higher than 0.05, we fail to reject the null hypothesis. In this regression model analysis, all variables have no statistical significant effect on household income.
#%%[markdown]
# Also from the boxplot about Household income vs public schools, we perform a ANOVA on it in order to learn whether there is any difference between areas whith different number of public schools. 
# 
# Before perform the ANOVA, we need to check if these four groups satisfy the normality and homoscedasticity. 
#%%
# Divide into four groups. 
bool1 = df4['public_school']==1
bool2 = df4['public_school']==2
bool3 = df4['public_school']==3
bool4 = df4['public_school']==4

dfsch1 = df4[bool1]
dfsch2 = df4[bool2]
dfsch3 = df4[bool3]
dfsch4 = df4[bool4]
# %%
sm.qqplot(dfsch1['Household_income'], fit=True, line='45')
plt.show()

sm.qqplot(dfsch2['Household_income'], fit=True, line='45')
plt.show()

sm.qqplot(dfsch3['Household_income'], fit=True, line='45')
plt.show()

sm.qqplot(dfsch4['Household_income'], fit=True, line='45')
plt.show()
#%%[markdown]
# From the qqplot above, most points are distributed aroud the red line, thus we assume they satisfy the normality. 
# %%
from scipy.stats import levene
p = levene(dfsch1['Household_income'],dfsch2['Household_income'],dfsch3['Household_income'],dfsch4['Household_income'])
print(p)
#%%[markdown]
# Null hypothesis: The variance for each groups is the same.   
# 
# Alternative hypothesis: At least one group has different variance with other groups. 
#
# Take $\alpha=0.05$, in the ANOVA table above, p-value is higher than 0.05. In that case, we fail to reject the null hypothesis. I draw the conclusion that they satisfy the homoscedasticity. Then we can do the ANOVA. 
# %%
print('ANOVA on Public_School')
anovsch = ols('Household_income~C(public_school)',data=df4).fit()
anovschin = anova_lm(anovsch, typ = 2)
print(anovschin)
#%%[markdown]
# Null hypothesis: The mean income for each groups is the same.   
# 
# Alternative hypothesis: At least one group has different mean income with other groups. 
#
# Take $\alpha=0.05$, in the ANOVA table above, p-value is higher than 0.05. In that case, we fail to reject the null hypothesis. I draw the conclusion that the number of public school would not have statistical significant effect on househould income. 







# %%
