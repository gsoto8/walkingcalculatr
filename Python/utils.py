## Python script (base) to extract walks from Location Based Services (LBS) data

import pandas as pd # data manipulation
import polars as pl
import os # addressing
import numpy as np # get angle
from path import Path # addressing
from math import radians, cos, sin, asin, sqrt, pi # dist calculations
import time
import re
import sys
from datetime import datetime as dt
import pyarrow # cast polars to data frame

from pygris import counties, tracts # county or tract level data
import geopandas as gpd
from shapely.plotting import plot_polygon
from shapely.ops import unary_union
from shapely import LineString, MultiPolygon, Point, Polygon, MultiLineString
import osmnx as ox
import matplotlib.pyplot as plt

# ACS population estimate extract for walk prevalence
from census import Census
from us import states

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) # Suppress run time warnings
warnings.filterwarnings("ignore", category=UserWarning) # Suppress user warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

####### Define Helper Functions #######

def concat_csv(dir):
    '''Combine all csvs in directory'''
    dfs = []
    for file in dir:
        temp = pd.read_csv(file)
        # temp = pl.read_csv(file, null_values = 'NA', truncate_ragged_lines=True)
        dfs.append(temp)
    return pd.concat(dfs)

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)

    Ref: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """

    try:

        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon1 - lon2 
        dlat = lat1 - lat2
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6378137 # Radius of earth in meters.
        return c * r
    
    except: 
        
        return 0

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two vectors 
    on the earth (specified in decimal degrees)
    """
       
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon1 - lon2 
    dlat = lat1 - lat2
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6378137 # Radius of earth in meters.
    return c * r

def RemoveStill(lats: list, lons: list, ids: list, accs: list):
    '''
    Returns a boolean array of non still points given numpy arrays (faster computation time).
    '''
    n = len(lats)
    start = 0
    ret = [False]*n
    for i in range(1, n): # start at second value
        
        if (ids[i]) != (ids[start]):
            start = i        
            continue
        
        # Calculate haversine distance between point i and start
        dist = haversine_distance(lats[i], lons[i], lats[start], lons[start]) # Update June 11
        # dist = haversine_distance(lons[i], lats[i], lons[start], lats[start])
        
        # If distance is smaller then combined accuracies from point i and start, classify as a still point
        if (dist < accs[i] + accs[start]):
            ret[i] = True
        else:
            start = i
        
    return ret

def RemoveJumps(df):
    '''
    Helper function to remove jump points in data.
    '''
    for i in range(3):
        df = df.filter(np.invert(makeLoRandomJump(df)))
        
    return df

def getAngle(dists: list, lats: list, lons: list):
    '''
    Calculates angle w.r.t. point B between three points (A, B, C)
    '''
    n = len(lats)
    ret = [0]*n
    ret[0] = np.nan
    ret[n-1] = np.nan

    # Calculate angle for each point
    for i in range(1, n - 1):
        ab = dists[i]
        bc = dists[i+1]
        ac = haversine_np(lons[i-1], lats[i-1], lons[i+1], lats[i+1])
        ret[i] = np.arccos((ab*ab + bc*bc - ac*ac)/(2*ab*bc))

    return np.array(ret)

def makeLoRandomJump(df):
    '''
    Find random jumps and remove them from data. 
    Returns a boolean array used in RemoveJumps()
    '''
    # Cast columns to numpy arrays
    try: lats = df['lat'].to_numpy()
    except: lats = df.select('lat').to_numpy()
    
    try: lons = df['lon'].to_numpy()
    except: lons = df.select('lat').to_numpy() # updated

    try: ids = df['userId'].to_numpy()
    except: ids = df.select('userId').to_numpy() # updated

    try: times = df['time'].to_numpy()
    except: times = df.select('time').to_numpy()   # updated

    # Calculate distance between current and previous point 
    lats_prev = np.roll(lats, 1)
    lons_prev = np.roll(lons, 1)
    lons_prev[0] = np.nan
    lats_prev[0] = np.nan
    dists = haversine_np(lons, lats, lons_prev, lats_prev)

    # Set all distances when device changes as null
    ids_next = np.append(ids[1:], np.nan)
    dists[ids != ids_next] = np.nan
    
    # Calculate velocities
    times_prev = np.roll(times, 1)
    times_prev[0] = 0

    vel = dists/(times - times_prev)
    vel_prev = np.roll(vel, 1) # update June 11

    vel[0] = np.nan # set first value to NaN to avoid overflow
    vel[ids != ids_next] = np.nan

    # Finds first point in series going faster than 200.
    vel_next = np.append(vel[1:], np.nan)

    # Find invalid speeds
    loIsFast = (vel_prev < 200) & (vel >= 200) # update June 11
    # loIsFast = (vel_next < 200) & (vel >= 200)

    # Calculate angle
    angles = getAngle(dists, lats, lons)
    
    # Find invalid angles
    loIsAngle = (vel > 5) & ((times - times_prev) < 300) & \
        (
            (angles < 0.25*pi) | (np.isnan(angles))
        )

    ret =  loIsFast | loIsAngle
    ret[np.isnan(ret)] = False

    return ret

def CleanData(df):
    '''
    Data cleaning functions.

    @df: Polars dataframe containing userId, time, lat and lon information
    '''
    print(df.columns)
    
    # Rename columns and sort data
    if ('id' in df.columns) and \
        ('latitude' in df.columns) and \
        ('longitude' in df.columns) and \
        ('unix_timestamp' in df.columns) and \
        ('horizontal_accuracy' in df.columns):

        dfclean = df.select(pl.col("id").str.to_lowercase().cast(pl.Categorical).alias('userId'),
                        pl.col('latitude').cast(pl.Float64).alias('lat'), 
                        pl.col('longitude').cast(pl.Float64).alias('lon'),
                        pl.col('unix_timestamp').alias('time'),
                        pl.col('horizontal_accuracy').cast(pl.Float64).alias('accuracy')
                        ).sort('userId', 'time', 'lat', 'lon',  'accuracy')
        # We want the timestamp to be 10 digits, so if the timestamp is 13 digits then we need to divide by 1000
        # UPDATE: check to see if timestamp is 10 digits before dividing by 1000
        #dfclean = dfclean.with_columns(time = (pl.col('time').cast(pl.Float64)/1000).cast(pl.Int32)) # Update time datatype and divide by 1000 if in miliseconds
        dfclean = dfclean.with_columns(
            pl.when(dfclean.with_columns(pl.col('time').cast(pl.String).str.len_chars() == 10))  # if 10 digits then do nothing
            .then(pl.col('time'))
            .when(dfclean.with_columns(pl.col('time').cast(pl.String).str.len_chars() == 13))
            .then((pl.col('time').cast(pl.Float64)/1000).cast(pl.Int32))                        # if 13 digits then divide by 1000
        )   
        dfclean = dfclean.unique(subset = ['userId', 'time'], maintain_order = True) # Drop duplicates

    elif ('ID' in df.columns) and \
        ('TIMESTAMP' in df.columns) and \
        ('LONGITUDE' in df.columns) and \
        ('TIMESTAMP' in df.columns) and \
        ('HORIZONTAL_ACCURACY' in df.columns):

        # Rename columns to match
        # UPDATE: TIMESTAMP instead of UNIX_TIMESTAMP
        dfclean = df.select(pl.col("ID").str.to_lowercase().cast(pl.Categorical).alias('userId'),
                        pl.col('LATITUDE').cast(pl.Float64).alias('lat'), 
                        pl.col('LONGITUDE').cast(pl.Float64).alias('lon'),
                        pl.col('TIMESTAMP'),
                        pl.col('HORIZONTAL_ACCURACY').cast(pl.Float64).alias('accuracy')
                        )
        
        # Convert UTC to Unix
        dfclean = dfclean.with_columns((pl.col('TIMESTAMP').map_elements(lambda x: dt.strptime(x + "+00:00","%Y-%m-%d %H:%M:%S%z").timestamp())).alias('time'))
        dfclean = dfclean.sort('userId', 'time', 'lat', 'lon',  'accuracy')
        
        dfclean = dfclean.with_columns(time = pl.col('time').cast(pl.Int32)) # Cast to int type
        dfclean = dfclean.unique(subset = ['userId', 'time'], maintain_order = True) # Drop duplicates

    else:

        # Rename columns and sort data
        dfclean = df.select(pl.col("userId").str.to_lowercase().cast(pl.Categorical),
                            pl.col('lat').cast(pl.Float64),
                            pl.col('lon').cast(pl.Float64),
                            pl.col('time'),
                            pl.col('accuracy').cast(pl.Float64)
                            ).sort('userId', 'time', 'lat', 'lon',  'accuracy')
        # We want the timestamp to be 10 digits, so if the timestamp is 13 digits then we need to divide by 1000
        # UPDATE: check to see if timestamp is 10 digits before dividing by 1000
        #dfclean = dfclean.with_columns(time = (pl.col('time').cast(pl.Float64)/1000).cast(pl.Int32)) # Update time datatype and divide by 1000 if in miliseconds
        dfclean = dfclean.with_columns(
            pl.when(dfclean.with_columns(pl.col('unix_timestamp').cast(pl.String).str.len_chars() == 10))  # if 10 digits then do nothing
            .then(pl.col('unix_timestamp'))
            .when(dfclean.with_columns(pl.col('unix_timestamp').cast(pl.String).str.len_chars() == 13))
            .then((pl.col('unix_timestamp').cast(pl.Float64)/1000).cast(pl.Int32))                        # if 13 digits then divide by 1000
        )
        dfclean = dfclean.unique(subset = ['userId', 'time'], maintain_order = True) # Drop duplicates

    # Remove inaccurate points
    dfclean = dfclean.filter(pl.col('accuracy') < 200) 
    dfclean = dfclean.filter((pl.col('lat') > 1) & (pl.col('lon') < -1)) 

    return dfclean.select('userId', 'time', 'lat', 'lon',  'accuracy')

def FindWalks(df, minSpeed: float = 0.5, maxSpeed: float= 5.0):
    '''
    Identify unique walks in data.

    @df: Polars dataframe containing userId, time, lat, lon, distance and angle information
    '''

    # Calculate distance between current and previous point 
    lats = df.select('lat').to_numpy().ravel()
    lats_prev = np.roll(lats, 1)
    lons = df.select('lon').to_numpy().ravel()
    lons_prev = np.roll(lons, 1)

    lons_prev[0] = np.nan
    lats_prev[0] = np.nan
    dists = haversine_np(lons, lats, lons_prev, lats_prev)

    # Set all distances when device changes to 0
    ids = df.select('userId').to_numpy().ravel()
    ids_next = np.append(ids[1:], np.nan)
    ids_prev = np.roll(ids, 1)
    ids_prev[0] = np.nan

    IsNewDevice = (ids != ids_prev)
    IsNextDevice = (ids != ids_next)
    dists[IsNextDevice] = np.nan
    
    # Calculate velocities
    times = df.select('time').to_numpy().ravel()
    vel = dists/(times - np.roll(times, 1))
    vel[0] = 0 # set first value to zero
    vel[IsNewDevice] = np.nan

    # Calculate angle
    angles = getAngle(dists, lats, lons)

    # Set first and last values to NAs
    angles[IsNewDevice | IsNextDevice] = 0

    # Find walks betwen minSpeed (in m/s) and maxSpeed
    isNewWalk = (vel < minSpeed) | (vel > maxSpeed) | (np.isnan(vel))
    isPrevWalk = np.roll(isNewWalk, 1)
    isPrevWalk[0] = True # set first to True
    isNextWalk = np.append(isNewWalk[1:], False) # set last to False

    # Assign temp variables
    df = df.with_columns(
        isNewWalk = isNewWalk,
        angle = angles,
        vel = vel,
        dist = dists)

    # Remove walks of length 2
    df_new = df.filter(np.invert(isPrevWalk & isNextWalk))

    # Remove walks of length 1
    isNewWalk = df_new.select('isNewWalk').to_numpy().ravel()
    isNextWalk = np.append(isNewWalk[1:], False) # set last to False 
    df_new = df_new.filter(np.invert(isNewWalk & isNextWalk))

    return df_new

def FindAcceptableWalks(df, minAngle: float = 15/180*pi, maxDist: float = 1500):
    '''
    Given walks identified in FindWalks(), find acceptable walks according to the walks' metrics (angle, dist, vel).

    @df: Polars dataframe containing userId, time, lat, lon, distance, velocity and angle information
    '''
        
    # Get indices when device changes
    ids = df.select('userId').to_numpy().ravel()
    ids_next = np.append(ids[1:], np.nan)
    ids_prev = np.roll(ids, 1)
    ids_prev[0] = np.nan

    IsNewDevice = (ids != ids_prev)
    IsNextDevice = (ids != ids_next)

    # Calculate distance between current and previous point 
    lats = df.select('lat').to_numpy().ravel()
    lats_prev = np.roll(lats, 1)
    lons = df.select('lon').to_numpy().ravel()
    lons_prev = np.roll(lons, 1)

    lons_prev[0] = np.nan
    lats_prev[0] = np.nan
    dists = haversine_np(lons, lats, lons_prev, lats_prev)

    # Get angle and velocity
    times = df.select('time').to_numpy().ravel()
    vel = dists/(times - np.roll(times, 1))
    vel[0] = 0 # set first value to zero
    vel[IsNewDevice] = np.nan
    angles = getAngle(dists, lats, lons)

    isNewWalk = df.select('isNewWalk').to_numpy().ravel()
    isNextWalk = np.append(isNewWalk[1:], True)

    # Set angle to 0 given conditions
    angles[
        np.isnan(angles) & 
        np.invert(isNewWalk) & \
        np.invert(IsNewDevice) & \
        np.invert(IsNextDevice)
    ] = 0

    # Find angles that are too small and account for nan values
    minAngle = 15/180*pi
    isAngleTooSmall = (angles < minAngle) 

    # Ignore null angles
    isAngleTooSmall[np.isnan(angles)] = False

    # Find walks that are too long
    maxDist = 1500
    isDistTooLong = (dists > maxDist) & np.invert(isNewWalk)

    # Also ignores first and last points in walk's angle, as that isn't apart of the walk. 
    # This line of code is separate to deal with tricky NA stuff. Basically some angles are NaN because they're at the start or end of a user, and don't have enough points to calculate. 
    # Those are fine and are dealt with by the next line, but angles are NAs for other reasons, so keep those marked as NA so that we can mark them as non-walks.
    isAngleTooSmall[isNewWalk | isNextWalk] = False

    # Give an ID for walks and non-walks -- reassign back to pandas dataframe
    df = df.with_columns(
        walkNum = np.cumsum(isNewWalk),
        isAngleTooSmall = isAngleTooSmall,
        isDistTooLong = isDistTooLong,
        isNewWalk = isNewWalk,
        angle = angles,
        vel = vel,
        dist = dists,
        )

    # Get valid walks
    df_valid = df.group_by('walkNum', maintain_order = True)\
                .agg(
                    isInWalk = ~(
                    (pl.col('isDistTooLong').any()) | \
                    (pl.col('isAngleTooSmall').sum() > 1) | \
                    ((pl.col('lat').count() == 3) & pl.col('isAngleTooSmall').any())  | \
                    (pl.col('isAngleTooSmall').is_null().any())
                    ))

    # Left join walks
    df_walk = df.join(df_valid, how = 'left', on = 'walkNum')

    # Remove non-valid walks
    df_walk = df_walk.filter(pl.col('isInWalk'))

    # Set beginning of walks dist, vel, and angle to NA and end of walks angle to NA
    df_walk = df_walk.with_columns(angle = pl.when(pl.col('isNewWalk')).then(np.nan).otherwise(pl.col('angle')),
                                    dist = pl.when(pl.col('isNewWalk')).then(np.nan).otherwise(pl.col('dist')),
                                    vel = pl.when(pl.col('isNewWalk')).then(np.nan).otherwise(pl.col('vel'))
                                    )

    return df_walk.select('userId', 'walkNum', 'lat', 'lon', 'time', 'accuracy', 'dist', 'vel', 'angle')

### Create ExtractWalks() wrapper function ##
def ExtractWalks(df):
    '''
    Wrapper function for all functions to calculate walks from a raw dataframe (polars) with userId, lat, lon and accuracy fields.
    Returns dataframe of walking pings and record counts for each step.
    '''
    record_counter = {}

    # Clean points
    clean_df = CleanData(df)
    record_counter['clean'] = clean_df.shape[0]

    # Remove still points
    ret = RemoveStill(clean_df.select('lat').to_numpy(),
                  clean_df.select('lon').to_numpy(),
                  clean_df.select('userId').to_numpy(),
                  clean_df.select('accuracy').to_numpy())
    df_remove = clean_df.filter(np.invert(ret))
    record_counter['remove still'] = df_remove.shape[0]

    # Remove jumps
    df_remove = RemoveJumps(df_remove)
    record_counter['remove jumps'] = df_remove.shape[0]

    # Remove still points again
    ret = RemoveStill(df_remove.select('lat').to_numpy(),
                  df_remove.select('lon').to_numpy(),
                  df_remove.select('userId').to_numpy(),
                  df_remove.select('accuracy').to_numpy())
    df_remove = df_remove.filter(np.invert(ret))
    record_counter['remove still again'] = df_remove.shape[0]

    # Calculate walks
    df_walk = FindWalks(df_remove)
    record_counter['find walks'] = df_walk.shape[0]

    # Get acceptable walks
    df_walk = df_walk.select('userId', 'lat', 'lon', 'time', 'accuracy', 'dist', 'vel' ,'angle', 'isNewWalk')
    df_final = FindAcceptableWalks(df_walk)

    return df_final, record_counter

### Walkable Areas ###
def fill_holes(multipolygon, eps: int = 200E-5):
    '''Removes all holes from a polygon with area < eps meters'''
    list_parts = []

    # Add if statement: polygon or multipolygon, break otherwise
    # Multi-polygon
    if isinstance(multipolygon, MultiPolygon):

        for polygon in multipolygon.geoms:
            list_interiors = []
            
            for interior in polygon.interiors:
                p = Polygon(interior)
                if p.area > eps:
                    list_interiors.append(interior)

            temp_pol = Polygon(polygon.exterior.coords, holes=list_interiors)
            list_parts.append(temp_pol)

    # Single polygon
    else:

        list_interiors = []
        for interior in multipolygon.interiors:
            p = Polygon(interior)
            if p.area > eps:
                list_interiors.append(interior)

        temp_pol = Polygon(multipolygon.exterior.coords, holes=list_interiors)
        list_parts.append(temp_pol)

    return unary_union(list_parts)

def get_walkable_areas(county_name: str = 'Ness', county_state: str = 'KS', year: int = 2023, fill_size: float = 200E-7):
    '''
    Calculate walkable shapefiles for all census tracts in a given county.

    @fill_size: Max size for holes to remove from polygon.
    '''
    cbsas = counties(state = county_state, cache = False, year = year)
    county_geo = cbsas[cbsas.NAME == county_name]

    # Get all roadways within a given polygon
    tags = {'highway': ['secondary', 'secondary_link', 'tertiary', 'tertiary_link']}  
    roadways = ox.features_from_polygon(county_geo.geometry.iat[0], tags)

    # Get all walking paths 
    tags = {'highway': ['residential', 'living_street', 'cycleway', 'pedestrian', 'footway', 'track', 'path']}
    local_streets = ox.features_from_polygon(county_geo.geometry.iat[0], tags)

    # Get all parks
    tags = {'leisure': ['park', 'playground', 'golf_course', 'nature_reserve']}
    parks = ox.features_from_polygon(county_geo.geometry.iat[0], tags)

    # Combine all walkable areas as shape file
    osm_cols = {'highway': 'property', 'leisure': 'property'}

    walkable_areas = pd.concat([
        roadways.loc[:,  (roadways.columns.isin(['name', 'highway', 'geometry']))].rename(columns = osm_cols),
        local_streets.loc[:,  (local_streets.columns.isin(['name',  'highway', 'geometry']))].rename(columns = osm_cols),
        parks.loc[:,  (parks.columns.isin(['name', 'leisure', 'geometry']))].rename(columns = osm_cols)
    ])

    # Get cartographic TIGER/Line file for tracts in a given state
    # Filter to only county tracts
    tracts_geos = tracts(state = county_state, cache = True, cb = True, year = year)
    county_tracts = tracts_geos[tracts_geos.NAMELSADCO == f'{county_name} County'].reset_index(drop = True)

    # Add buffer geometry around each census tract (500 m)
    county_tracts['buffer_geometry'] = county_tracts['geometry'].buffer(500E-5)
    county_tracts['buffer_geometry'] = county_tracts['buffer_geometry'].to_crs('epsg:4326')
    walkable_areas = walkable_areas.to_crs('epsg:4326')

    # Get intersection with geometry and all linestrings
    tract_geos = []
    tract_names = []
    tract_states = []
    tract_geoms = []

    for i in range(county_tracts.shape[0]):

        # Get intersection with geometry and all walkable areas (EPSG 4326)
        geo_poly = county_tracts['buffer_geometry'].iat[i]
        walkable_areas['intersect_geometry'] = walkable_areas.apply(lambda x: geo_poly.intersection(x.geometry), axis = 1)
        walkable_areas['buffer_intersect_geometry'] = walkable_areas['intersect_geometry'].buffer(20E-5)
        walkable_areas['buffer_intersect_geometry'] = walkable_areas['buffer_intersect_geometry']

        # Combine all geometries into one multipolygon
        walk_geom = unary_union(walkable_areas['buffer_intersect_geometry'][~walkable_areas['buffer_intersect_geometry'].is_empty].to_list())

        # Fill small holes
        walk_geom_simple = fill_holes(walk_geom, eps = (fill_size)**2)

        # Save to list for df
        tract_geos.append(county_tracts['GEOID'].iat[i])
        tract_states.append(county_tracts['STUSPS'].iat[i])
        tract_names.append(county_tracts['NAMELSAD'].iat[i])
        tract_geoms.append(walk_geom_simple)

    # Save to summary df
    county_tracts_walkable_areas = gpd.GeoDataFrame(geometry=tract_geoms)
    county_tracts_walkable_areas.insert(0, 'GEOID', tract_geos)
    county_tracts_walkable_areas.insert(1, 'Name',  tract_names)
    county_tracts_walkable_areas.insert(2, 'State', tract_states)

    # Save to geojson
    try: county_tracts_walkable_areas = county_tracts_walkable_areas.to_crs('epsg:4326')
    except: county_tracts_walkable_areas = county_tracts_walkable_areas.set_crs('epsg:4326')

    return county_tracts_walkable_areas

def get_walkable(walks_df, walkable_areas):
    '''
    Find all walking pings given all census tract walkable shapefiles created in get_walkable_areas().

    Returns a dataframe with all walking pings.
    
    Note:
    For ~1 million records, 30min exec time
    For ~1.3 million, 45 min exec time
    '''
    # Find walks in walkable areas with each tract
    point_geos = walks_df.apply(lambda x: Point(x['lon'], x['lat']), axis = 1)
    walks_gpd = gpd.GeoDataFrame(walks_df,  geometry = point_geos, crs = 'epsg:4326')

    # Iterate through each census tract / walkable area
    N = walks_gpd.shape[0]
    is_walkable = np.repeat([False], N)
    for idx in walkable_areas.index:

        geom = walkable_areas.loc[idx, 'geometry']
        geoids  = walkable_areas.loc[idx, 'GEOID']

        # Get all pings that are walkable
        walkable_pings = walks_gpd.within(geom).to_numpy()

        # Keep track of whether a ping is walkble
        is_walkable = (is_walkable) | (walkable_pings)

    walks_gpd['is_walkable'] = is_walkable

    # Get percent walkable by walk number
    pct_walkable_by_walk = walks_gpd.groupby('walkNum').apply(lambda x: (x.is_walkable.sum())/x.shape[0])

    # Save final walks in walkable table to memory
    walks_in_walkable = walks_gpd.merge(pct_walkable_by_walk.reset_index()).rename(columns = {0: 'pct_walkable', 'walkN': 'walkNum'})\
    [['userId', 'walkNum', 'lat', 'lon', 'time', 'accuracy', 'dist', 'vel', 'angle', 'pct_walkable']]

    return walks_in_walkable


########### Auxiliary Functions ##############

def county_sidewalk_density(county_name,
                            state,
                            year,
                            ped_accessible_highway_keys = ["trunk", "primary_link",
                                                            "living_street", "secondary",
                                                            "secondary_link", "tertiary",
                                                            "tertiary_link"],
                            sidewalk_keys = ["pedestrian", "footway",
                                            "path", "residential", "track"]):
    
     # Get cartographic TIGER/Line file for given county in the U.S.
     all_counties = counties(cache = False, cb = True, year = year)
     county = all_counties[(all_counties.NAME == county_name) & (all_counties.STUSPS == state)].geometry.iat[0]

     # Get sidewalk data (E. Moro values)
     osm_sidewalks = ox.features_from_polygon(county, {'highway': sidewalk_keys})

     # Get pedestrian accessible highway data
     osm_highways = ox.features_from_polygon(county, {'highway': ped_accessible_highway_keys})
     
     # Get line data
     osm_sidewalk_lines = osm_sidewalks[osm_sidewalks.apply(lambda x: isinstance(x.geometry, LineString) or isinstance(x.geometry, MultiLineString), axis = 1)]
     osm_highway_lines = osm_highways[osm_highways.apply(lambda x: isinstance(x.geometry, LineString) or isinstance(x.geometry, MultiLineString), axis = 1)]
     
     # Cast to meter crs: https://epsg.io/3786
     osm_sidewalk_lines = osm_sidewalk_lines.to_crs(3786)
     osm_highway_lines = osm_highway_lines.to_crs(3786)

     # Get side walk length from linestring using st_length()
     osm_sidewalk_lines['osm_line_length'] = osm_sidewalk_lines.apply(lambda x: x.geometry.length, axis = 1)
     osm_sidewalks_km = sum(osm_sidewalk_lines['osm_line_length']/1000)

     osm_highway_lines['osm_line_length'] = osm_highway_lines.apply(lambda x: x.geometry.length, axis = 1)
     osm_highway_km = sum(osm_highway_lines['osm_line_length']/1000)

     # Calculate sidewalk density for given county
     osm_density = osm_sidewalks_km/osm_highway_km

     summary_stats = pd.DataFrame(
          zip(["county_name", "sidewalk_length_km","ped_accessible_highway_length_km", "sidewalk_density"],
               [county_name, osm_sidewalks_km, osm_highway_km, osm_density]
          ), columns = ['metric', 'value']
     )

     return summary_stats

def tract_sidewalk_density(county_name = "San Francisco", 
                        state = "CA", 
                        year = 2022,
                        ped_accessible_highway_keys = ["trunk", "primary_link",
                                                        "living_street", "secondary",
                                                        "secondary_link", "tertiary",
                                                        "tertiary_link"],
                        sidewalk_keys = ["pedestrian", "footway",
                                        "path", "residential", "track"],
                        download_tracts = True,
                        OSM_cache_loc = './test_path'):
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    county_tracts = tracts(county = county_name, state = state, cache = False, year = year, cb = True)

    # Create directory for files if does not exist
    if not(os.path.exists(OSM_cache_loc)):

        Path(OSM_cache_loc).mkdir() # make path
        print("No OSM cached data location was provided or is invalid. OSM data will be extracted and cached to ", OSM_cache_loc)

    elif (not(os.path.exists(OSM_cache_loc)) and not(download_tracts)): print("Using saved OSM data from ", OSM_cache_loc)

    else: print("Caching OSM data into ", OSM_cache_loc)

    # Save sidewalk and highway OSM for each tract
    tract_dir = Path(OSM_cache_loc)

    if download_tracts:

        for i, tract in enumerate(county_tracts.geometry):

            GEOID = county_tracts['GEOID'].iat[i]
            
            try:

                # Get sidewalk data (E. Moro values)
                osm_sidewalks = ox.features_from_polygon(tract, {'highway': sidewalk_keys})

                # Get pedestrian accessible highway data
                osm_highways = ox.features_from_polygon(tract, {'highway': ped_accessible_highway_keys})

                # Get line data
                osm_sidewalk_lines = osm_sidewalks[osm_sidewalks.apply(lambda x: isinstance(x.geometry, LineString) or isinstance(x.geometry, MultiLineString), axis = 1)]
                osm_highway_lines = osm_highways[osm_highways.apply(lambda x: isinstance(x.geometry, LineString) or isinstance(x.geometry, MultiLineString), axis = 1)]
                
                # Cast to meter crs: https://epsg.io/3786
                osm_sidewalk_lines = osm_sidewalk_lines.to_crs(3786)
                osm_highway_lines = osm_highway_lines.to_crs(3786)

                # Save to local as GeoJSON
                gpd.GeoDataFrame(osm_sidewalk_lines.reset_index()[['osmid', 'highway', 'geometry']]).to_file(tract_dir / f'{GEOID}_sidewalk.geojson', driver = "GeoJSON")
                gpd.GeoDataFrame(osm_highway_lines.reset_index()[['osmid', 'highway', 'geometry']]).to_file(tract_dir / f'{GEOID}_highway.geojson', driver = "GeoJSON")

            except: continue
            
    osm_sidewalks_tract_lens = []
    osm_highway_tract_lens = []
    geoids = []

    for i, tract in enumerate(county_tracts.geometry):
        
        GEOID = county_tracts['GEOID'].iat[i]
        geoids.append(GEOID)
        
        try:
            
            # Load OSM data for each census tract file
            osm_sidewalk_lines = gpd.read_file(tract_dir / f'{GEOID}_sidewalk.geojson')
            osm_highway_lines = gpd.read_file(tract_dir / f'{GEOID}_highway.geojson')

            # Get side walk length from linestring using st_length()
            osm_sidewalks_km = sum(osm_sidewalk_lines.apply(lambda x: x.geometry.length, axis = 1)/1000)
            osm_highway_km = sum(osm_highway_lines.apply(lambda x: x.geometry.length, axis = 1)/1000)

            # Calculate sidewalk density for given county
            osm_sidewalks_tract_lens.append(osm_sidewalks_km)
            osm_highway_tract_lens.append(osm_highway_km)
        
        except:
            
            # Assign as -1 if unable to calculate
            osm_sidewalks_tract_lens.append(-1)
            osm_highway_tract_lens.append(-1)

    # Create summary dataframe
    tract_summary_stats = pd.DataFrame()
    tract_summary_stats['GEOID'] = geoids
    tract_summary_stats['sidewalk_length_km'] = osm_sidewalks_tract_lens
    tract_summary_stats['ped_accessible_highway_length_km'] = osm_highway_tract_lens
    tract_summary_stats['sidewalk_density'] = tract_summary_stats['sidewalk_length_km'] / tract_summary_stats['ped_accessible_highway_length_km']

    # Sort by descending sidewalk density
    tract_summary_stats = tract_summary_stats.sort_values(by = 'sidewalk_density', ascending=False)
    tract_summary_stats = tract_summary_stats.merge(county_tracts[['GEOID', 'NAMELSAD', 'geometry']], how = 'left')
    tract_summary_stats = tract_summary_stats[['GEOID', 'NAMELSAD', 'geometry', 'sidewalk_length_km',
                                            'ped_accessible_highway_length_km','sidewalk_density']]
    
    return tract_summary_stats

#### Walk Prevalence Functions
def getCountyTracts(county_name, state, year):
    '''
    Returns a GeoDataFrame of TIGRIS shapefiles for all census tracts 
    in a given U.S. county county_name.
        
        Parameters:
            county_name (str): U.S. county name (e.g., "Suffolk")
            state (str): U.S. county state (e.g., "MA")
            year (int): Valid year from the 5-Year American Community Survey (e.g., 2022)
        
        Returns:
            countyTracts (geoDataFrame): GeoDataFrame of TIGRIS shapefiles 
                                        for all census tracts in county_name
    '''

    # Get cartographic TIGER/Line file for tracts in a given state
    # Filter to only county tracts
    countyTracts = tracts(county = county_name, state = state, year = year)

    # Add buffer geometry around each census tract (500 m)
    countyTracts['buffer_geometry'] = countyTracts['geometry'].buffer(500E-5)
    countyTracts['buffer_geometry'] = countyTracts['buffer_geometry'].to_crs('epsg:4326')

    return countyTracts

def getAPIpopulation(county_name, state, year, censusAPIkey):
    '''
    Returns a GeoDataFrame of the ACS tract-level population estimates from the Census API.
    To get a valid API key, sign up here: https://api.census.gov/data/key_signup.html).
        
        Parameters:
            county_name (str): U.S. county name (e.g., "Suffolk")
            state (str): U.S. county state (e.g., "MA")
            year (int): Valid year from the 5-Year American Community Survey (e.g., 2022)
            censusAPIkey (str): Census API key from https://api.census.gov/data/key_signup.html
        
        Returns:
            tractPopulation (GeoDataFrame): Merged GeoDataFrame containing the ACS population estimates
                                            and TIGRIS shapefiles for each census tract in a given county_name.
    '''
    countyTracts = getCountyTracts(county_name=county_name, state = state, year = year)

    # valid census API key necessary to extract information https://api.census.gov/data/key_signup.html
    c = Census(censusAPIkey)
    state_fip = states.mapping('abbr', 'fips')[state]
    # try:
    acs_df = c.acs5.state_county_tract(fields = ('NAME', 'B01003_001E'),
                                        state_fips = state_fip,
                                        county_fips = "*",
                                        tract = "*",
                                        year = year)

    tractPopulation = pd.DataFrame(acs_df).rename(columns={'B01003_001E':'pop_estimate'})

    # create GEOID from FIPS
    tractPopulation["GEOID"] = tractPopulation["state"] + tractPopulation["county"] + tractPopulation["tract"]

    # split NAME string into separate parts
    tractPopulation[['NAMELSAD', 'COUNTY', 'STATE']] = tractPopulation['NAME'].str.split("; ", expand=True)
    tractPopulation = tractPopulation.drop(columns=['state', 'county', 'tract', 'NAME'])

    # merge county_nametracts and population
    tractPopulation = countyTracts[['GEOID','geometry']].merge(tractPopulation, on = 'GEOID')

    # Convert to geodataframe and project to 4326
    tractPopulation = gpd.GeoDataFrame(tractPopulation).to_crs('epsg:4326')

    return tractPopulation

# add function that grabs cached version of population counts (2022)
def getCachepopulation(county_name, state, year, ACSpath):
    '''
    Returns a DataFrame of the ACS tract-level population estimates for a given U.S. county.
    Population estimates from the 2022 5-Year American Community Survey are included in package.
        
        Parameters:
            county_name (str): U.S. county name (e.g., "Suffolk")
            state (str): U.S. county state (e.g., "MA")
            year (int): Valid year from the 5-Year American Community Survey (e.g., 2022)
            ACSpath (str): Valid file path of the saved ACS population estimates file.
                            Must contain GEOID (or ability to extract one), population estimates, 
                            and county, state, and census tract name
        
        Returns:
            cachePopulation (GeoDataFrame): Merged GeoDataFrame containing the ACS population estimates
                                            and TIGRIS shapefiles for each census tract in a given county_name.
    '''
    # If error in getACSpopulation, then use default cached version
    cachePopulation = pd.read_csv(ACSpath,
                                dtype={'GEOID': str}
                                ).rename(columns={'estimate':'pop_estimate'})

    # split NAME string into separate parts
    cachePopulation[['NAMELSAD', 'COUNTY', 'STATE']] = cachePopulation['NAME'].str.split("; ", expand=True)

    cachePopulation = cachePopulation.drop(columns=['NAME', 'variable', 'moe'])

    countyTracts = getCountyTracts(county_name=county_name, state = state, year = year)

    # merge county tracts and population
    cachePopulation = countyTracts[['GEOID','geometry']].merge(cachePopulation, on = 'GEOID')

    # Convert to geodataframe and project to 4326
    cachePopulation = gpd.GeoDataFrame(cachePopulation).to_crs('epsg:4326')
    
    return cachePopulation

# get ACS population and join to county tract geometry
def getACSpopulation(county_name, state, year, censusAPIkey = None, ACSpath = '../data-raw/US_tract_population2022.csv.gz'):
    '''
    Returns a DataFrame of the ACS tract-level population estimates for a given U.S. county.
    Population estimates from the 2022 5-Year American Community Survey are included in package
    or from the Census API for the given year.
        
        Parameters:
            county_name (str): U.S. county name (e.g., "Suffolk")
            state (str): U.S. county state (e.g., "MA")
            year (int): Valid year from the 5-Year American Community Survey (e.g., 2022)
            censusAPIkey (str): Optional Census API key from https://api.census.gov/data/key_signup.html.
                                If API key is not provided, then function will return cached 
                                2022 ACS population estimates.
            ACSpath (str): Valid file path of the saved ACS population estimates file.
                            Must contain GEOID (or ability to extract one), population estimates, 
                            and county, state, and census tract name. Default uses cached 2022
                            ACS population estimates in /data-raw
        
        Returns:
            tractPopulation (GeoDataFrame): Merged GeoDataFrame containing the ACS population estimates
                                            and TIGRIS shapefiles for each census tract in a given county_name.
    '''
    
    if censusAPIkey is None:
        print("No Census API key provided, using cached 2022 ACS population data")
        tractPopulation = getCachepopulation(county_name= county_name, state = state, year = year, ACSpath = ACSpath)
    else:
        print("Using provided Census API key value from censusAPIkey")
        try:
            tractPopulation = getAPIpopulation(county_name= county_name, state = state, year = year, censusAPIkey = censusAPIkey)
            return tractPopulation
        except:
            print("Census API key or invalid year error, using cached 2022 ACS population data")
            tractPopulation = getCachepopulation(county_name= county_name, state = state, year = year, ACSpath = ACSpath)
    return tractPopulation



# map walks to census tracts
def intersectWalks(walkableWalksDF, tractPopulation):
    '''
    Returns a DataFrame mapping LBS pings from walks in walkable areas to the census tract GEOID that
    the ping is located in.
        
        Parameters:
            walkableWalksDF (DataFrame): pandas DataFrame of walks extracted from LBS data and mapped to walkable areas.
            tractPopulation (GeoDataFrame): Merged GeoDataFrame containing the ACS population estimates
                                            and TIGRIS shapefiles for each census tract in a given county_name.
        
        Returns:
            walkIntersection (GeoDataFrame): Merged GeoDataFrame containing the ACS population estimates, GEOID, and
                                            shapefiles of the census tracts that intersect each walk ping.
    '''
    
    # filter to only fractions that are 0.75
    walkableWalksDF = walkableWalksDF[walkableWalksDF['fraction'] > 0.75].reset_index(drop=True)

    # create geometry from coordinates
    walkableWalksDF['geometry'] = walkableWalksDF.apply(lambda x: Point(x['lon'], x['lat']), axis = 1)
    walkableWalksGDF = gpd.GeoDataFrame(walkableWalksDF, crs = 'epsg:4326')

    walkIntersection = gpd.sjoin(walkableWalksGDF, tractPopulation[['GEOID','NAMELSAD','pop_estimate', 'geometry']],predicate='intersects').reset_index(drop=True)
    
    return walkIntersection

def tractWalkPrevalence(walkableWalksDF, county_name, state, year, censusAPIkey = None, ACSpath = '../data-raw/US_tract_population2022.csv.gz'):
    '''
    Returns a DataFrame of the walk prevalence, population estimate, and number of walks in walkable areas for
    each census tract in a given U.S. county.
        
        Parameters:
            walkableWalksDF (DataFrame): pandas DataFrame of walks extracted from LBS data and mapped to walkable areas.
            county_name (str): U.S. county name (e.g., "Suffolk")
            state (str): U.S. county state (e.g., "MA")
            year (int): Valid year from the 5-Year American Community Survey (e.g., 2022)
            censusAPIkey (str): Optional Census API key from https://api.census.gov/data/key_signup.html.
                                If API key is not provided, then function will return cached 
                                2022 ACS population estimates.
            ACSpath (str): Valid file path of the saved ACS population estimates file.
                            Must contain GEOID (or ability to extract one), population estimates, 
                            and county, state, and census tract name. Default uses cached 2022
                            ACS population estimates in /data-raw
        
        Returns:
            walkPrevDF (DataFrame): DataFrame of walk prevalence, population estimates, and number of walks for each
            census tract GEOID in a given U.S. county_name.
    '''
    # get tract population
    tractPopulation = getACSpopulation(county_name= county_name, state = state, year = year, censusAPIkey=censusAPIkey, ACSpath = ACSpath)
    
    # removing walks in non-walkable areas
    walkableWalksDF = walkableWalksDF[walkableWalksDF['fraction'] > 0.75].reset_index(drop=True)

    # find intersecting areas and remove duplicates
    intersectDF = intersectWalks(walkableWalksDF, tractPopulation).drop_duplicates()

    # group by GEOID and pop_estimate, then sum number of walks
    # intersect_df['walkNum'] = intersect_df['walkNum'].apply(str) # convert to string
    intersectDF[['userId', 'walkNum']] = intersectDF[['userId', 'walkNum']].astype(str)

    intersectDF['count_walkable'] = intersectDF['userId'] + intersectDF['walkNum'] # create unique ID for walk count
    walkPrevDF = intersectDF.groupby(['GEOID','NAMELSAD', 'pop_estimate'])[['count_walkable']].nunique().reset_index()

    # Divide number of walks with number of tract pop
    walkPrevDF['walk_prev'] = walkPrevDF['count_walkable'] / walkPrevDF['pop_estimate']

    # Left join tract pop to walk prev df to determine which census tracts don't have walks
    walkPrevDF = tractPopulation.merge(walkPrevDF, how='left', on = ['GEOID', 'NAMELSAD', 'pop_estimate'])
    
    # If pop == 0 and count_walkable > 0, then NA
    walkPrevDF.loc[(walkPrevDF['count_walkable'] > 0) & (walkPrevDF['pop_estimate'] == 0), 'walk_prev'] = "No Residents"

    # If count walkable == 0, then count_walkable and walk prev are 0
    walkPrevDF.loc[pd.isna(walkPrevDF['count_walkable']), ['count_walkable', 'walk_prev']] = 0

    # remove all rows that have 0 walks and 0 residents
    walkPrevDF = walkPrevDF[~((walkPrevDF['count_walkable'] == 0) & (walkPrevDF['pop_estimate'] == 0))].reset_index(drop=True)
    # reorder columns
    walkPrevDF = walkPrevDF[['GEOID', 'NAMELSAD', 'geometry', 'count_walkable', 'pop_estimate', 'walk_prev']]

    return walkPrevDF

# get county level walk prevalence
def countyWalkPrevalence(walkableWalksDF, county_name,state, year, censusAPIkey = None, ACSpath = '../data-raw/US_tract_population2022.csv.gz'):
    '''
    Returns a DataFrame of the walk prevalence, population estimate, and number of walks in walkable areas at a county level.
        
        Parameters:
            walkableWalksDF (DataFrame): pandas DataFrame of walks extracted from LBS data and mapped to walkable areas.
            county_name (str): U.S. county name (e.g., "Suffolk")
            state (str): U.S. county state (e.g., "MA")
            year (int): Valid year from the 5-Year American Community Survey (e.g., 2022)
            censusAPIkey (str): Optional Census API key from https://api.census.gov/data/key_signup.html.
                                If API key is not provided, then function will return cached 
                                2022 ACS population estimates.
            ACSpath (str): Valid file path of the saved ACS population estimates file.
                            Must contain GEOID (or ability to extract one), population estimates, 
                            and county, state, and census tract name. Default uses cached 2022
                            ACS population estimates in /data-raw
        
        Returns:
            countyWalkPrevDF (DataFrame): DataFrame of walk prevalence, population estimates, and number of walks for the overall
            county_name.
    '''

    walkableWalksDF = walkableWalksDF[walkableWalksDF['fraction'] > 0.75].reset_index(drop=True)
    tractPopulation = getACSpopulation(county_name= county_name, state = state, year = year, censusAPIkey = censusAPIkey, ACSpath = ACSpath)

    # total population of all tracts
    popCount = sum(tractPopulation['pop_estimate'])

    # count number of unique walks
    countWalkable = walkableWalksDF.groupby(['userId', 'walkNum']).ngroups

    # county level walk prevalence
    countyWalkPrev = countWalkable/popCount

    countyWalkPrevDF = pd.DataFrame(data = {'count_walkable':[countWalkable], 'pop_estimate':[popCount], 'walk_prev':[countyWalkPrev]})
    return countyWalkPrevDF

def walkPrevalenceSummary(tractWalkPrevalenceDF):
    '''
    Returns a summary of metrics for tract-level walk prevalence found using tractWalkPrevalence().
        
        Parameters:
            tractWalkPrevalenceDF (DataFrame): pandas DataFrame of walk prevalence by county census tract.
            Found using tractWalkPrevalence().

        Returns:
            countyWalkPrevDF (DataFrame): DataFrame summary of mean, standard deviation, median, min, max, and total for
                                        count of walks in walkable areas, population estimates, and walk prevalence across
                                        all census tracts in a given county.
    '''
    # select only the columns that need summarizing
    walkPrevSummary = tractWalkPrevalenceDF[['count_walkable', 'pop_estimate', 'walk_prev']]
    
    # transform dataframe to long format
    walkPrevSummary = pd.melt(walkPrevSummary)

    # Remove all rows with 'No Resident' as the value, then convert to numeric
    walkPrevSummary = walkPrevSummary[pd.to_numeric(walkPrevSummary['value'], errors='coerce').notnull()].reset_index(drop=True)

    # Summarize using aggregate metrics
    walkPrevSummary = walkPrevSummary.groupby('variable').agg(
        Mean=('value', np.mean),
        SD=('value', np.std),
        Median=('value', np.median),
        Min=('value', min),
        Max=('value', max),
        Total=('value', np.sum)).reset_index()

    # Since 'Total' value is not needed for walk_prev, replace with 'None'
    walkPrevSummary.loc[walkPrevSummary['variable'] == 'walk_prev', 'Total'] = None

    return walkPrevSummary

