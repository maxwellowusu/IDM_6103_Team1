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
plt.style.use('ggplot')
from scipy.stats import iqr
import scipy.stats as stats
from scipy.stats import normaltest
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
df = gpd.read_file('./data/dataset.csv')
print(df.head())
df = df.fillna(0)

#%%
#This is only for Evelyn since some of the packages for the geojson ds are not working in her Mac
df = pd.read_csv('./data/dataset.csv') #I am using the csv file that's why I commented the other one
df = df.fillna(0)
print(df.head())
#%%
df1 = df
df1=df1.fillna(df1.mean())
df1.info()
df1.isnull().sum().sum() #0: checking for null values
#%%
df1.shape # 179 rows and 20 columns
print(df1.columns.values) # since I had to use the csv file I wanted to verify the variables
df1.dtypes
df1.describe


#%%
#EDA SMART QUESTION 1 
#Is there segregation in public investment in D.C.?
#Descriptive statistics for bus stops. Do the same for the rest of the variables
df1.bus_stops.mean()
df1.bus_stops.median()
df1.bus_stops.mode()
df1.bus_stops.std()
df1.bus_stops.var()
iqr(df1['bus_stops'])
print(df1.bus_stops.skew())
#%% 
#Histogram 
fig, axs = plt.subplots(2, 3, figsize=(10, 10))
sns.histplot(data=df1, x="bus_stops", kde=True, color="skyblue", ax=axs[0, 0])
sns.histplot(data=df1, x="public_school", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=df1, x="metro_station", kde=True, color="gold", ax=axs[0, 2])
sns.histplot(data=df1, x="Black_population", kde=True, color="teal", ax=axs[1, 0])
sns.histplot(data=df1, x="White_population", kde=True, color="purple", ax=axs[1, 1])
plt.show()

#%%  
#Scatterplot 
sns.pairplot(df1, y_vars=['bus_stops','public_school', 'metro_station'], y_vars=['Black_population', 'White_population'],
            hue='Population', height=1.5)
plt.show()
#%% 
#Looking at the correlation
df_corr = df[['bus_stops', 'Black_population', 'White_population','public_school', 'metro_station']].dropna().corr()
df_corr
#%% 
#map of correlation
sns.heatmap(df_corr, annot=True)
#END OF EDA SMART QUESTION 1 
#DISTRIBUTIONS AND TESTS SMART Q1
#Normal distribution for bus stops
def check_p_val(p_val, alpha):
 if p_val < alpha:
   print('We have evidence to reject the null hypothesis.')
 else:
   print('We do not have evidence to reject the null hypothesis.')
#%%
#Plotting normal distribution bus stops
xs = np.arange(df1.bus_stops.min(), df1.bus_stops.max(), 0.1)
fit = stats.norm.pdf(xs, np.mean(df1.bus_stops), np.std(df1.bus_stops))
plt.plot(xs, fit, label='Normal Dist.', lw=3)
plt.hist(df1.bus_stops, 50, density=True, label='Actual Data');
plt.legend()
#%%
#testing distribution
stat, p_val = normaltest(df1.bus_stops)
print('\nNormaltest p-value is: {:1.2f} \n'.format(p_val))
check_p_val(p_val, alpha=0.05)
#%%
#Normal distribution for public schools
def check_p_val(p_val, alpha):
 if p_val < alpha:
   print('We have evidence to reject the null hypothesis.')
 else:
   print('We do not have evidence to reject the null hypothesis.')
#%% 
#Plotting normal distribution public schools
xp = np.arange(df1.public_school.min(), df1.public_school.max(), 0.1)
fit1 = stats.norm.pdf(xp, np.mean(df1.public_school), np.std(df1.public_school))
plt.plot(xp, fit1, label='Normal Dist.', lw=3)
plt.hist(df1.public_school, 50, density=True, label='Actual Data')
plt.legend()
#%%
#testing distribution
stat, p_val = normaltest(df1.public_school)
print('\nNormaltest p-value is: {:1.2f} \n'.format(p_val))
check_p_val(p_val, alpha=0.05)
#%%
#Normal distribution for metro stations
def check_p_val(p_val, alpha):
 if p_val < alpha:
   print('We have evidence to reject the null hypothesis.')
 else:
   print('We do not have evidence to reject the null hypothesis.')
#%%
#Normal distribution for metro stations
xq = np.arange(df1.metro_station.min(), df1.metro_station.max(), 0.1)
fit2 = stats.norm.pdf(xq, np.mean(df.metro_station), np.std(df1.metro_station))
plt.plot(xq, fit2, label='Normal Dist.', lw=3)
plt.hist(df1.metro_station, 50, density=True, label='Actual Data');
plt.legend();
#%%
#testing distribution
stat, p_val = normaltest(df1.metro_station)
print('\nNormaltest p-value is: {:1.2f} \n'.format(p_val))
check_p_val(p_val, alpha=0.05)
#%%
# ztest
# black population bus stops 
zScore, pValue = ztest(df1['Black_population'], df1['bus_stops'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# white population bus stops 
zScore, pValue = ztest(df1['White_population'], df1['bus_stops'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# black population metro stations
zScore, pValue = ztest(df1['Black_population'], df1['metro_station'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# white population metro stations
zScore, pValue = ztest(df1['White_population'], df1['metro_station'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# black population 
zScore, pValue = ztest(df1['Black_population'], df1['public_school'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# white population metro stations
zScore, pValue = ztest(df1['White_population'], df1['public_school'])
print('zscore:', zScore, 'p-value:', pValue)
#END OF DISTRIBUTIONS AND TESTS SMART Q1
#%% 
#EDA SMART QUESTION 2
#Does public infrastructure reduce green areas? 
#Descriptive statistics
df1.tree.mean()
df1.tree.median()
df1.tree.mode()
df1.tree.std()
df1.tree.var()
iqr(df1['tree'])
print(df1.tree.skew())
#%% 
df1.park.mean()
df1.park.median()
df1.park.mode()
df1.park.std()
df1.park.var()
iqr(df1['park'])
print(df1.park.skew())
#%% 
#Histogram 
fig, axs = plt.subplots(2, 3, figsize=(10, 10))
sns.histplot(data=df1, x="bus_stops", kde=True, color="skyblue", ax=axs[0, 0])
sns.histplot(data=df1, x="public_school", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=df1, x="metro_station", kde=True, color="gold", ax=axs[0, 2])
sns.histplot(data=df1, x="tree", kde=True, color="teal", ax=axs[1, 0])
sns.histplot(data=df1, x="park", kde=True, color="purple", ax=axs[1, 1])
plt.show()
#%%    
#Scatterplot
sns.pairplot(df1, y_vars=['park', 'tree'], x_vars=['bus_stops','public_school', 'metro_station'],
            hue='Population', height=1.5)
plt.show()

#%% 
#Looking at the correlation
df1_corr2 = df1[['bus_stops', 'tree', 'park','public_school', 'metro_station']].dropna().corr()
df1_corr2
#%% 
sns.heatmap(df1_corr2, annot=True)
#END OF EDA SMART QUESTION 2
#DISTRIBUTIONS AND TESTS SMART Q2
#%%
def check_p_val(p_val, alpha):
 if p_val < alpha:
   print('We have evidence to reject the null hypothesis.')
 else:
   print('We do not have evidence to reject the null hypothesis.')
#%%
#Normal distribution for tree
xt = np.arange(df1.tree.min(), df1.tree.max(), 0.1)
fitt = stats.norm.pdf(xt, np.mean(df1.tree), np.std(df1.tree))
plt.plot(xt, fitt, label='Normal Dist.', lw=3)
plt.hist(df1.tree, 50, density=True, label='Actual Data');
plt.legend()
#%%
#testing distribution
stat, p_val = normaltest(df1.tree)
print('\nNormaltest p-value is: {:1.2f} \n'.format(p_val))
check_p_val(p_val, alpha=0.05)
#%%
def check_p_val(p_val, alpha):
 if p_val < alpha:
   print('We have evidence to reject the null hypothesis.')
 else:
   print('We do not have evidence to reject the null hypothesis.')
#%%
#Normal distribution for park
x = np.arange(df1.park.min(), df1.park.max(), 0.1)
fit = stats.norm.pdf(x, np.mean(df1.park), np.std(df1.park))
plt.plot(x, fit, label='Normal Dist.', lw=3)
plt.hist(df1.tree, 50, density=True, label='Actual Data')
plt.legend()
#%%
#testing distribution
stat, p_val = normaltest(df1.park)
print('\nNormaltest p-value is: {:1.2f} \n'.format(p_val))
check_p_val(p_val, alpha=0.05)

#This is the graph I used for park variable because the previous code didn't give me a good plot
sns.displot(df1.park, kde=True)
#%%
# ztest
# tree bus stops 
zScore, pValue = ztest(df1['tree'], df1['bus_stops'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# park bus stops 
zScore, pValue = ztest(df1['park'], df1['bus_stops'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# tree  metro stations
zScore, pValue = ztest(df1['tree'], df1['metro_station'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# park metro stations
zScore, pValue = ztest(df1['park'], df1['metro_station'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# tree public school
zScore, pValue = ztest(df1['tree'], df1['public_school'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# park public school
zScore, pValue = ztest(df1['park'], df1['public_school'])
print('zscore:', zScore, 'p-value:', pValue)
#EDA SMART QUESTION 3
#Is there a relationship between distribution of infrastructure, socioeconomic, and race groups?
#Descriptive statistics
df1.Household_income.mean()
df1.Household_income.median()
df1.Household_income.mode()
df1.Household_income.std()
df1.Household_income.var()
iqr(df1['Household_income'])
print(df1.Household_income.skew())
#%% 
#Histogram 
fig, axs = plt.subplots(2, 3, figsize=(10, 10))
sns.histplot(data=df1, x="bus_stops", kde=True, color="skyblue", ax=axs[0, 0])
sns.histplot(data=df1, x="public_school", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=df1, x="metro_station", kde=True, color="gold", ax=axs[0, 2])
sns.histplot(data=df1, x="tree", kde=True, color="teal", ax=axs[1, 0])
sns.histplot(data=df1, x="park", kde=True, color="purple", ax=axs[1, 1])
sns.histplot(data=df1, x="Household_income", kde=True, color="pink", ax=axs[1, 2])
plt.show()
#%%    
#Scatterplot
sns.pairplot(df1, y_vars=['park', 'tree', 'Household_income'], x_vars=['bus_stops','public_school', 'metro_station'],
            hue='Population', height=1.5)
plt.show()

#%% 
#Looking at the correlation
df1_corr2 = df1[['bus_stops', 'tree', 'Household_income','park','public_school', 'metro_station']].dropna().corr()
df1_corr2
#%% 
sns.heatmap(df1_corr2, annot=True)
#END OF EDA SMART QUESTION 2
#DISTRIBUTIONS AND TESTS SMART Q2
#%%
def check_p_val(p_val, alpha):
 if p_val < alpha:
   print('We have evidence to reject the null hypothesis.')
 else:
   print('We do not have evidence to reject the null hypothesis.')
#%%
#Normal distribution for tree
xt = np.arange(df1.tree.min(), df1.tree.max(), 0.1)
fitt = stats.norm.pdf(xt, np.mean(df1.tree), np.std(df1.tree))
plt.plot(xt, fitt, label='Normal Dist.', lw=3)
plt.hist(df1.tree, 50, density=True, label='Actual Data');
plt.legend();
#%%
#testing distribution
stat, p_val = normaltest(df1.tree)
print('\nNormaltest p-value is: {:1.2f} \n'.format(p_val))
check_p_val(p_val, alpha=0.05)
#%%
def check_p_val(p_val, alpha):
 if p_val < alpha:
   print('We have evidence to reject the null hypothesis.')
 else:
   print('We do not have evidence to reject the null hypothesis.')
#%%
#Normal distribution for park
x = np.arange(df1.park.min(), df1.park.max(), 0.1)
fit = stats.norm.pdf(x, np.mean(df1.park), np.std(df1.park))
plt.plot(x, fit, label='Normal Dist.', lw=3)
plt.hist(df1.tree, 50, density=True, label='Actual Data')
plt.legend()
#%%
#testing distribution
stat, p_val = normaltest(df1.park)
print('\nNormaltest p-value is: {:1.2f} \n'.format(p_val))
check_p_val(p_val, alpha=0.05)

#This is the graph I used for park variable because the previous code didn't give me a good plot
sns.displot(df1.park, kde=True)
#%%
# ztest
# tree bus stops 
zScore, pValue = ztest(df1['tree'], df1['bus_stops'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# park bus stops 
zScore, pValue = ztest(df1['park'], df1['bus_stops'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# tree  metro stations
zScore, pValue = ztest(df1['tree'], df1['metro_station'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# park metro stations
zScore, pValue = ztest(df1['park'], df1['metro_station'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# tree public school
zScore, pValue = ztest(df1['tree'], df1['public_school'])
print('zscore:', zScore, 'p-value:', pValue)
#%%
# park public school
zScore, pValue = ztest(df1['park'], df1['public_school'])
print('zscore:', zScore, 'p-value:', pValue)












#%% [markdown]
# Is there segregation in public infrastructure investment in the DC?

# plot Bus stops 
fig, axes = plt.subplots(figsize=(10, 10), sharex=True)  
axes.set_title('Bus stops counts per census tracts') 
df.plot(ax=axes, column= 'bus_stops', scheme='NaturalBreaks', k=5, \
            cmap='YlOrRd', legend=True,
            legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
         )
plt.savefig('./output/bus_stops.jpeg') 
plt.show()
  
#%%
# Trees
fig, axes = plt.subplots(figsize=(10, 10), sharex=True)  
axes.set_title('Tree counts per census tract') 
df.plot(ax=axes, column= 'tree', scheme='NaturalBreaks', k=5, \
            cmap='YlOrRd', legend=True,
            legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
         )
plt.savefig('./output/trees.jpeg')   
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
   outfile = f'./output/{variable}.jpeg'
   plt.savefig(outfile)
   plt.show()

   
   return None
#%%
# Public schools
plot_infrastructure('public_school', 'Public school counts per census tracts', 'No Public Schools')

#%%
# Metro stations
plot_infrastructure('metro_station','Metro stations count per census tracts', 'No Metro Station')

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
plt.savefig('./output/race.jpeg') 
plt.show()

# Visually there is seems to be clusters of race in DC. 

#%%
# Household income
fig, axes = plt.subplots(figsize=(10, 10), sharex=True)  
axes.set_title('Household Income') 
df.plot(ax=axes, column= 'Household_income', scheme='NaturalBreaks', k=5, \
            cmap='YlOrRd', legend=True,
            legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'interval': True},
         )
plt.savefig('./output/Household_income.jpeg') 
plt.show()


#%%
df1 =df.dropna()
# Logistic model 
x = df1[['Household_income', 'bus_stops', 'metro_station','tree']]
y = df1['Race']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1 )

#%%
# logistric regression
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()  # instantiate
logit.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', logit.score(x_test, y_test))
print('Logit model accuracy (with the train set):', logit.score(x_train, y_train))
#%%
# df1['lg_pred'] = logit.predict(x)


#%%

#%%
# Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, random_state = 0)

rf.fit (x_train, y_train)
print('Random forest model accuracy (with the test set):', rf.score(x_test, y_test))
print('Random forest model accuracy (with the train set):', rf.score(x_train, y_train))
#%%
# df1['rf_pred'] = rf.predict(x)
#%%

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
#%%[markdown]
## Smart Question 2
# Do regression for question 2: Does public infrastructure reduce green areas? 
import statsmodels.api as sm
fit2=sm.formula.ols('tree~Household_income+bus_stops+public_school+metro_station+park',data = df).fit()
print(fit2.summary())

#%%[markdown]
## Smart Question 3
# Is there a relationship between distribution of infrastructure and socio-economic  and race groups?
#%%[markdown]
# In order to do some data process, define a new dataframe. 
# Calculate the ration of white_population to Total Population, and add it to a new column named race_ratio. 
#%%
df3 = df
df3["race_ratiowhite"]=df3[["White_population","Population"]].apply(lambda x:x["White_population"]/x["Population"],axis=1)
df3["race_ratioblack"]=df3[["Black_population","Population"]].apply(lambda x:x["Black_population"]/x["Population"],axis=1)
df3["race_ratiothers"]=df3[["race_ratiowhite","race_ratioblack"]].apply(lambda x:1-x["race_ratiowhite"]-x["race_ratioblack"],axis=1)
df3.head()
#%%
# Build regression model between income and white poputlation: 
import statsmodels.api as sm
fit_income31=sm.formula.ols('Household_income~race_ratiowhite',data = df3).fit()
print(fit_income31.summary())
#%%
# Build regression model between income and black poputlation: 
import statsmodels.api as sm
fit_income32=sm.formula.ols('Household_income~race_ratioblack',data = df3).fit()
print(fit_income32.summary())
#%%
# Build regression model between income and other poputlation: 
import statsmodels.api as sm
fit_income33=sm.formula.ols('Household_income~race_ratiothers',data = df3).fit()
print(fit_income33.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is less than 0.05, we reject the null hypothesis in favor of alternative hypothesis. The race ratio is negative ralated to household income. That as the white population increse, the household income would decrease. 
#%%
# Build regression model between bus stops and white poputlation: 
import statsmodels.api as sm
fit_bus31=sm.formula.ols('bus_stops~race_ratiowhite',data = df3).fit()
print(fit_bus31.summary())
#%%
# Build regression model between bus stops and black poputlation: 
import statsmodels.api as sm
fit_bus32=sm.formula.ols('bus_stops~race_ratioblack',data = df3).fit()
print(fit_bus32.summary())
#%%
# Build regression model between bus stops and other poputlation: 
import statsmodels.api as sm
fit_bus33=sm.formula.ols('bus_stops~race_ratiothers',data = df3).fit()
print(fit_bus33.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is higher than 0.05, we fail to reject the null hypothesis. The race ratio would not have statistical significant effect on bus stops. 
#%%
# Build regression model between public schools and white poputlation: 
import statsmodels.api as sm
fit_school31=sm.formula.ols('public_school~race_ratiowhite',data = df3).fit()
print(fit_school31.summary())
#%%
# Build regression model between public schools and black poputlation: 
import statsmodels.api as sm
fit_school32=sm.formula.ols('public_school~race_ratioblack',data = df3).fit()
print(fit_school32.summary())
#%%
# Build regression model between public schools and other poputlation: 
import statsmodels.api as sm
fit_school33=sm.formula.ols('public_school~race_ratiothers',data = df3).fit()
print(fit_school33.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is higher than 0.05, we fail to reject the null hypothesis. The race ratio would not have statistical significant effect on the number of public schools. 
#%%
# Build regression model between metro stations and white poputlation: 
import statsmodels.api as sm
fit_metro31=sm.formula.ols('metro_station~race_ratiowhite',data = df3).fit()
print(fit_metro31.summary())
#%%
# Build regression model between metro stations and black poputlation: 
import statsmodels.api as sm
fit_metro32=sm.formula.ols('metro_station~race_ratioblack',data = df3).fit()
print(fit_metro32.summary())
#%%
# Build regression model between metro stations and other poputlation: 
import statsmodels.api as sm
fit_metro33=sm.formula.ols('metro_station~race_ratiothers',data = df3).fit()
print(fit_metro33.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is less than 0.05, we reject the null hypothesis in favor of alternative hypothesis. The race ratio is positive ralated to metro station. That as the white population increse, the metro station would also increse. 
#%%
# Build regression model between trees and white poputlation: 
import statsmodels.api as sm
fit_tree31=sm.formula.ols('tree~race_ratiowhite',data = df3).fit()
print(fit_tree31.summary())
#%%
# Build regression model between trees and black poputlation: 
import statsmodels.api as sm
fit_tree32=sm.formula.ols('tree~race_ratioblack',data = df3).fit()
print(fit_tree32.summary())
#%%
# Build regression model between trees and other poputlation: 
import statsmodels.api as sm
fit_tree33=sm.formula.ols('tree~race_ratiothers',data = df3).fit()
print(fit_tree33.summary())
#%%[markdown]
# Null hypothesis: The coefficient of variables in regression model equals to 0.  
# 
# Alternative hypothesis: The coefficient of variables in regression model does not equal to 0. 
# 
# Take $\alpha=0.05$, the p-values of race ratio is higher than 0.05, we fail to reject the null hypothesis. The race ratio would not have statistical significant effect on the number of trees. 

#%%
# Build regression model between parks and white poputlation: 
import statsmodels.api as sm
fit_park31=sm.formula.ols('park~race_ratiowhite',data = df3).fit()
print(fit_park31.summary())
#%%
# Build regression model between parks and black poputlation: 
import statsmodels.api as sm
fit_park32=sm.formula.ols('park~race_ratioblack',data = df3).fit()
print(fit_park32.summary())
#%%
# Build regression model between parks and other poputlation: 
import statsmodels.api as sm
fit_park33=sm.formula.ols('park~race_ratiothers',data = df3).fit()
print(fit_park33.summary())
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
df4.head()
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
# However, take $\alpha=0.05$, the p-values of public_school and park are higher than 0.05, we fail to reject the null hypothesis. The p-values of public_school and park are higher than 0.05, we fail to reject the null hypothesis. The p-values of bus_stops, metro_station and tree are less than 0.05, we reject the null hypothesis in favor of alternative hypothesis. 
# 
# In this regression model analysis, as bus_stops increases in 1 degree with other variables remain the same, the household income would increase in 9.1913 degree. In this regression model analysis, as metro_station increases in 1 degree with other variables remain the same, the household income would decrease in 10.9027 degree. In this regression model analysis, as tree increases in 1 degree with other variables remain the same, the household income would decrease in 0.0381 degree. 
#%%[markdown]
# Also from the boxplot about Household income vs public schools, we perform a ANOVA on it in order to learn whether there is any difference between areas with different number of public schools. 
# 
# Before perform the ANOVA, we need to check if these four groups satisfy the normality and homoscedasticity. 
#%%
# Divide into five groups. 
bool0 = df4['public_school']==0
bool1 = df4['public_school']==1
bool2 = df4['public_school']==2
bool3 = df4['public_school']==3
bool4 = df4['public_school']==4

dfsch0 = df4[bool0]
dfsch1 = df4[bool1]
dfsch2 = df4[bool2]
dfsch3 = df4[bool3]
dfsch4 = df4[bool4]
# %%
sm.qqplot(dfsch0['Household_income'], fit=True, line='45')
plt.show()

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
p = levene(dfsch0['Household_income'],dfsch1['Household_income'],dfsch2['Household_income'],dfsch3['Household_income'],dfsch4['Household_income'])
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
