#!/usr/bin/env python
# coding: utf-8

#%% [markdown]

# ## Project Data Preparation

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
# | C17002_001E | Ratio of income to poverty level in the past 12 months

#%% 
# !pip install census
# %%

# import libraries
from census import Census
from us import states
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import seaborn as sns
sns.set_theme(style='darkgrid')

#%%

#%%
# Download street from open data DC

bus_stops = gpd.read_file('https://opendata.arcgis.com/datasets/e85b5321a5a84ff9af56fd614dab81b3_53.geojson')
print(bus_stops.shape)
bus_stops.head(2)

#%%
ax = bus_stops.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
ctx.add_basemap(ax, crs=bus_stops.crs)

#%%

# Download schools from open data DC

public_schools = gpd.read_file('https://opendata.arcgis.com/datasets/4ac321b2d409438ebd76a6569ad94034_5.geojson')
print(public_schools.shape)
public_schools.head(2)

#%%
# Download trees from open data DC

trees = gpd.read_file('https://opendata.arcgis.com/datasets/f6c3c04113944f23a7993f2e603abaf2_23.geojson')
print(trees.shape)
trees.head(2)

# Download metro station from open data DC

metro_station = gpd.read_file('https://opendata.arcgis.com/datasets/ab5661e1a4d74a338ee51cd9533ac787_50.geojson')
print(metro_station.shape)
metro_station.head(2)

# # Download parks and recreation from open data DC

parks = gpd.read_file('https://opendata.arcgis.com/datasets/287eaa2ecbff4d699762bbc6795ffdca_9.geojson')
print(parks.shape)
parks.head(2)


############################ end of GIS data download ###############
#%%
# Download census data 
session = Census("808b8bdd29d3424881a13740265bdf2c3d7b0980") # Please get a Census API key from https://api.census.gov/data/key_signup.html
# list of variables to download census data
census_vars = ['B01003_001E','B02001_002E','B02001_003E', 'B25074_020E']
# download variables based on their name, geography and year
dc_census = session.acs5.state_county_tract(fields = ('NAME', *census_vars ),
                                      state_fips = states.DC.fips,
                                      county_fips = "*",
                                      tract = "*",
                                      year = 2019)


#%%
# convert the data into a dataframe and view head()
dc_df = pd.DataFrame(dc_census)

# rename variables 
variable_dict = {'B01003_001E': 'Population','B02001_002E':'White_population', 'B02001_003E':'Black_population', 'B25074_020E': 'Household_income'}
dc_df.rename(columns=variable_dict, inplace=True)
dc_df.head()

#%%
# wget to download tiger shapefile (DC census tract)
# ! wget https://www2.census.gov/geo/tiger/TIGER2017/TRACT/tl_2017_11_tract.zip 

#%%
# import geopandas and read in the census tract file 

# make sure to upzip the file manual. I don't know how to do it in the script yet. 

dc_shp = gpd.read_file('./tl_2017_11_tract/tl_2017_11_tract.shp')
dc_shp.to_crs("EPSG:32618", inplace=True)
# remove unneeded columns
dc_shp.drop(columns=['NAMELSAD','MTFCC','FUNCSTAT','ALAND','AWATER','INTPTLAT','INTPTLON'], inplace=True)
dc_shp.head(2)

#%%
# merge census data with census tract file

#first generate the GEOID for census data
# Create variable GEOID for census variables
dc_df.loc[:,"GEOID"] = dc_df["state"] + dc_df["county"] + dc_df["tract"]
dc_df["GEOID"] 

# merge data (tiger shapefile and census variables) 
dc_join = dc_shp.merge(right=dc_df, how='left',on='GEOID', validate='one_to_one')
dc_join.head()


# %%
dc_join.plot()
# %%

# # Lets join the seleted infrastructure to the dataframe.

# join census data and street trees shapefile
# First, we spatial join the two data and group the number of infrastructure points (count) to the census tract
bus_stops.to_crs("EPSG:32618", inplace=True)
bus_stop = gpd.sjoin(dc_join,
                         bus_stops,
                         how = "inner",
                         predicate="intersects")
bus_stop.head(2)
bus_stop['bus_stops'] = 1
bus_stop = bus_stop.groupby("GEOID").sum()
bus_stop = bus_stop[['bus_stops']]
bus_stop.head()

#%%
#merge groupby and DC_census track
bus_stop_merge = dc_join.merge(right=bus_stop, how='left',on='GEOID', validate='one_to_one')
bus_stop_merge.head()

# %%
# public_schools
public_schools.to_crs("EPSG:32618", inplace=True)
public_school = gpd.sjoin(dc_join,
                         public_schools,
                         how = "inner",
                         predicate="intersects")
public_school.head(2)
public_school['public_school'] = 1
public_school = public_school.groupby("GEOID").sum()
public_school = public_school[['public_school']]
public_school.head()

#%%
#merge groupby and DC_census track
bus_sch_merge = bus_stop_merge.merge(right=public_school, how='left',on='GEOID', validate='one_to_one')
bus_sch_merge.head()

# %%
# metro_station
metro_station.to_crs("EPSG:32618", inplace=True)
metro_stations = gpd.sjoin(dc_join,
                         metro_station,
                         how = "inner",
                         predicate="intersects")
metro_stations.head(2)
metro_stations['metro_station'] = 1
metro_stations = metro_stations.groupby("GEOID").sum()
metro_stations = metro_stations[['metro_station']]
metro_stations.head()

# %%
#merge groupby and DC_census track
bus_sch_metro_merge = bus_sch_merge.merge(right=metro_stations, how='left',on='GEOID', validate='one_to_one')
bus_sch_metro_merge.head()

# %%
# trees

trees.to_crs("EPSG:32618", inplace=True)
tree = gpd.sjoin(dc_join,
                         trees,
                         how = "inner",
                         predicate="intersects")
tree.head(2)
tree['tree'] = 1
tree = tree.groupby("GEOID").sum()
tree = tree[['tree']]
tree.head()

# %%
#merge groupby and DC_census track
bus_sch_metro_trees_merge = bus_sch_metro_merge.merge(right=tree, how='left',on='GEOID', validate='one_to_one')
bus_sch_metro_trees_merge.head()

# %%
# parks

parks.to_crs("EPSG:32618", inplace=True)
park = gpd.sjoin(dc_join,
                         parks,
                         how = "inner",
                         predicate="intersects")
park.head(2)
park['park'] = 1
park = park.groupby("GEOID").sum()
park = park[['park']]
park.head()

# %%

#merge groupby and DC_census track
bus_sch_metro_trees_parks_merge = bus_sch_metro_trees_merge.merge(right=park, how='left',on='GEOID', validate='one_to_one')
bus_sch_metro_trees_parks_merge.head()
# %%
outfile = 'dataset.geojson'
bus_sch_metro_trees_parks_merge.to_file(outfile, driver='GeoJSON')
# %%
output_csv = 'dataset.csv'
bus_sch_metro_trees_parks_merge.to_csv(output_csv, sep=',', header=True)
# %%
