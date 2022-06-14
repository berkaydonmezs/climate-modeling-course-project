import cartopy
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
#import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import proplot
#import pyproj
import rioxarray
import xarray as xr


# VARIABLE NAME MAP DICTIONARIES

model_field_map = {'t2m': 'T_2M', 'prec': 'TOT_PREC',
                   't': 'T', 'relhum': 'RELHUM',
                   'geopotential': 'FI'}

era_field_map = {'t2m': 't2m', 'prec': 'tp',
                 't': 't', 'relhum':'r', 'geopotential':'z'}

graph_field_map = {'t2m': '2 M Temperature', 'prec': 'Total Precipitation',
                   't850': 'Temperature 850 hPa', 't925': 'Temperature 925 hPa',
                   'relhum850': 'Relative Humidity 850 hPa', 'relhum925': 'Relative Humidity 925 hPa',
                   'geopotential850': 'Geopotential Height 850 hPa',
                   'geopotential925': 'Geopotential Height 925 hPa'}

unit_map = {'relhum': '%',
            't': 'K',
            't2m': 'K',
            'prec': 'mm'}


# PROJECTION RELATED FUNCTIONS

class ModelProj():
    
    def __init__(self):
        
        self.pole_longitude = -147.0
        self.pole_latitude = 50.0
        
        # crs info for the model
        self.crs_info_model = cartopy.crs.RotatedPole(pole_longitude=self.pole_longitude, 
                                                     pole_latitude=self.pole_latitude, 
                                                     central_rotated_longitude=0).proj4_params

def regrid_match(da_to_match, da_to_be_matched, da_to_match_crs, da_to_be_matched_crs):
    """
    Regrid a file grid to a target grid. Requires input data array
    
    Return target file and regridded file
    
    """
    
    # set crs for the target grid
    da_to_match = da_to_match.rio.write_crs(da_to_match_crs)
    da_to_match = da_to_match.rio.set_spatial_dims(x_dim='rlon', y_dim='rlat')
    
    # set crs for the file for which regridding will be performed
    da_to_be_matched = da_to_be_matched.rio.write_crs(da_to_be_matched_crs)
    da_to_be_matched = da_to_be_matched.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
    
    
    da_to_be_matched = da_to_be_matched.rio.reproject_match(da_to_match).rename({'y':'rlat', 'x':'rlon', })
    
    
    return da_to_match, da_to_be_matched


# INDEX CALCULATION FUNCTIONS
        
def num_days_temp_above_thresh(daily_tmax, threshold):
    """
    Number of days where daily maximum temperature is above a threshold
    """
    num_data = xr.where(daily_tmax>threshold, 1, 0).sum(dim='time')  
    return num_data

def num_days_precip_below_thresh(daily_p, threshold):
    """
    Number of days where daily precipitation is below a threshold (mm)
    """
    num_data = xr.where(daily_p<threshold, 1, 0).sum(dim='time')  
    return num_data
        

# FUNCTIONS FOR SAVING DATA
    
def save_score_file(ds, var, level, prepath):
    
    if level:
        file_name = fr'model_{var}_{level}_scores.nc'
    else:
        file_name = fr'model_{var}_scores.nc'
        
    ds.to_netcdf(prepath + file_name)
    return file_name + " created"

def save_anom_file(ds, prepath, anom_type):
    file_name = fr'era_{anom_type}_anomaly.nc'
    ds.to_netcdf(prepath + file_name)
    return file_name + " created"

def save_verif_file(da, var, level, prepath, verif_type):
    da.name = var
    
    if level:
        file_name = fr'model_{var}_{level}_{verif_type}_verif.nc'
    else:
        file_name = fr'model_{var}_{verif_type}_verif.nc'
        
    da.to_netcdf(prepath + file_name)
    return file_name + " created"
        

# FUNCTIONS FOR PROCESSING DATA

def aggregate_file(da, agg_by, agg_for='6H'):
    """
    Aggregate file on the time dimension (6H by fefault)
    
    """
    da_aggr = da.resample(time=agg_for)
    
    if agg_by=='sum':
        da_aggr = da_aggr.sum()
        
        print(fr"Warning: aggregating by {agg_for} sum")
        
        
    elif agg_by=='mean':
        da_aggr = da_aggr.mean()
        
        print(fr"Warning: aggregating by {agg_for} mean")
    
    elif agg_by=='max':
        da_aggr = da_aggr.max()
        
        print(fr"Warning: aggregating by {agg_for} max")
    
    elif agg_by=='min':
        da_aggr = da_aggr.min()
        
        print(fr"Warning: aggregating by {agg_for} min")
        
    else:
        raise Exception("Sorry, the only aggregation procedures supported are sum, mean, max, and min")
        
    return da_aggr

def get_era_climatology(ds, start_date, end_date, cli_type='month'):
    """
    Get the groupbyed climatology of the ERA5 data
    """
    
    ds_slice = ds.sel(time=slice(fr'{start_date}', fr'{end_date}'))
    return ds_slice.groupby(fr'time.{cli_type}').mean()


# FUNCTIONSFOR RETRIEVING DATA

def get_post_processed_anomaly_files(anom_type, prepath):
    """
    Get the post processed score files
    
    return anom_pp
    
    """
    
    pp_path = prepath + fr'era_{anom_type}_anomaly.nc'
        
    anom_pp = xr.open_dataset(pp_path)
    return anom_pp

def get_post_processed_score_files(var_type, var, level, prepath):
    """
    Get the post processed score files
    
    return score_pp
    
    """
    if level:
        pp_path = prepath + fr'model_{var}_{level}_scores.nc'
    else:
        pp_path = prepath + fr'model_{var}_scores.nc'
        
    score_pp = xr.open_dataset(pp_path)
    return score_pp

def get_post_processed_verif_files(verif_type, var_type, var, level, prepath):
    """
    Get the post processed verification files
    
    return da_pp
    
    """
    if level:
        pp_path = prepath + fr'model_{var}_{level}_{verif_type}_verif.nc'
    else:
        pp_path = prepath + fr'model_{var}_{verif_type}_verif.nc'
        
    da_pp = xr.open_dataset(pp_path)
    return da_pp
    
def get_model_era_files(var_type, var, level, start_date, end_date):
    """
    Get the ERA5 (hourly) and model (6-H) files
    
    return da_model, da_era
    
    """
    
    era_path = fr"data/era5/era5_{var_type}_hourly_project_data_climate_modeling.nc"
    model_path = fr"data/model/model_{var_type}_6h.nc"
    
    var_era = era_field_map[var]
    var_model = model_field_map[var]
    
    
    da_era = xr.open_dataset(era_path)[var_era].sel(time=slice(start_date, end_date))
    da_model = xr.open_dataset(model_path)[var_model].sel(time=slice(start_date, end_date))
    
    if level:
        da_era = da_era.sel(level=level)
        da_model = da_model.sel(pressure=level*100) # hPa to Pa
        
        
    return da_model, da_era
        

# FUNCTIONS FOR PLOTTING
    
def plot_verif_map(data_df, var_short, cmap, vmin, vmax, norm, ticks,
                   crs_data, graphic_no, var_name,
                    method, difference_method, fig_array, unit):
    
    # graphic features
    cmap = cmap
    vmin = vmin
    vmax = vmax
    norm = norm
    ticks = ticks


    # projection
    crs_data = crs_data

    # Create Figure -------------------------
    fig, axs = proplot.subplots(fig_array, 
                              aspect=10, axwidth=5, proj=crs_data,
                              hratios=tuple(np.ones(len(fig_array), dtype=int)),
                              includepanels=True, hspace=-2.3, wspace=0.15)

    for i in range(graphic_no):
        axs[i].format(lonlim=(24, 45), latlim=(34, 42),
                      labels=False, longrid=False, latgrid = False)


    # türkiye harici shapeler
    # Find the China boundary polygon.
    shpfilename = shpreader.natural_earth(resolution='10m',
                                                  category='cultural',
                                                  name='admin_0_countries')

    cts = ['Syria', 'Iraq', 'Iran', 'Azerbaijan', 'Armenia',
           'Russia', 'Georgia', 'Bulgaria', 'Greece', 'Cyprus',
           'Northern Cyprus', 'Turkey', 'Albania', 'North Macedonia',
           'Montenegro', 'Serbia', 'Italy']


    for country in shpreader.Reader(shpfilename).records():
        if country.attributes['ADMIN'] in cts:
            count_shp = country.geometry

            for i in range(graphic_no):
                axs[i].add_geometries([count_shp], cartopy.crs.PlateCarree(),
                                          facecolor='none', edgecolor = 'black',
                                          linewidth = 0.5, zorder = 2.2,)
    
    # graphic
    for i in range(graphic_no):
        mesh = axs[i].pcolormesh(data_df['lon'], data_df['lat'],
                             data_df[i], norm = norm,
                             cmap = cmap, vmin = vmin, vmax = vmax,
                             zorder = 2.1)

    # CMAP ----------------------

    cbar = fig.colorbar(mesh, ticks=ticks, loc='b', drawedges = False, shrink=1, space = -1.1, aspect = 30, )
    cbar.ax.tick_params(labelsize=11, )
    cbar.set_label(label='{} ({}) | {}'.format(var_name, unit, difference_method), size=16, loc = 'center', y=0.35, weight = 'bold')
    cbar.outline.set_linewidth(2)
    cbar.minorticks_off()
    cbar.ax.get_children()[4].set_color('black')
    cbar.solids.set_linewidth(1)


    # TEXT
    # Build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    
    title_method = data_df.dims[0]
    for i in range(graphic_no):
        
        if method=='Month':
            axs[i].set_title(r'{}: {}'.format(method, data_df[i][title_method].dt.month.values), fontsize = 12,
                         loc = 'left', pad = -14, y = 0.01, x=0.020, weight = 'bold',)
        elif method=='Season':
            axs[i].set_title(r'{}: {}'.format(method, data_df[i][title_method].season.values), fontsize = 12,
                         loc = 'left', pad = -14, y = 0.01, x=0.020, weight = 'bold',)
        
    # savefig    
    plt.savefig(fr'pics/report/{method.lower()}_verif_{var_short}_graph.jpeg',
                bbox_inches='tight', optimize=False, progressive=True, dpi=300)
    
    
def plot_score_map(data_df, var_short, cmap, vmin, vmax, norm, ticks,
                   crs_data, graphic_no, var_name,
                    method, difference_method, fig_array, unit):
    
    # graphic features
    cmap = cmap
    vmin = vmin
    vmax = vmax
    norm = norm
    ticks = ticks


    # projection
    crs_data = crs_data

    # Create Figure -------------------------
    fig, axs = proplot.subplots(fig_array, 
                              aspect=10, axwidth=5, proj=crs_data,
                              hratios=tuple(np.ones(len(fig_array), dtype=int)),
                              includepanels=True, hspace=-1.40, wspace=0.15)

    for i in range(graphic_no):
        axs[i].format(lonlim=(24, 45), latlim=(34, 42),
                      labels=False, longrid=False, latgrid = False)


    # türkiye harici shapeler
    # Find the China boundary polygon.
    shpfilename = shpreader.natural_earth(resolution='10m',
                                                  category='cultural',
                                                  name='admin_0_countries')

    cts = ['Syria', 'Iraq', 'Iran', 'Azerbaijan', 'Armenia',
           'Russia', 'Georgia', 'Bulgaria', 'Greece', 'Cyprus',
           'Northern Cyprus', 'Turkey', 'Albania', 'North Macedonia',
           'Montenegro', 'Serbia', 'Italy']


    for country in shpreader.Reader(shpfilename).records():
        if country.attributes['ADMIN'] in cts:
            count_shp = country.geometry

            for i in range(graphic_no):
                axs[i].add_geometries([count_shp], cartopy.crs.PlateCarree(),
                                          facecolor='none', edgecolor = 'black',
                                          linewidth = 0.5, zorder = 2.2,)
    
    # graphic
    for i in range(graphic_no):
        mesh = axs[i].pcolormesh(data_df['lon'], data_df['lat'],
                             data_df['rmse'], norm = norm,
                             cmap = cmap, vmin = vmin, vmax = vmax,
                             zorder = 2.1)

    # CMAP ----------------------

    cbar = fig.colorbar(mesh, ticks=ticks, loc='b', drawedges = False, shrink=1, space = -1.2, aspect = 50, )
    cbar.ax.tick_params(labelsize=11, )
    cbar.set_label(label='{} | {}'.format(var_name, difference_method), size=16, loc = 'center', y=0.35, weight = 'bold')
    cbar.outline.set_linewidth(2)
    cbar.minorticks_off()
    cbar.ax.get_children()[4].set_color('black')
    cbar.solids.set_linewidth(1)


    # TEXT
    # Build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    
    for i in range(graphic_no):
        
        
        axs[i].set_title(r'{}: {}'.format(method, 'rmse'), fontsize = 12, color='white',
                         loc = 'left', pad = -14, y = 0.01, x=0.020, weight = 'bold',)
        
    # savefig   
    plt.savefig(fr'pics/report/{method.lower()}_rmse_{var_short}_graph.jpeg',
                bbox_inches='tight', optimize=False, progressive=True, dpi=300)
    
def plot_anom_map(data_df, var_short, cmap, vmin, vmax, norm, ticks,
                   crs_data, graphic_no, var_name,
                    method, difference_method, fig_array, unit):
    
    # graphic features
    cmap = cmap
    vmin = vmin
    vmax = vmax
    norm = norm
    ticks = ticks


    # projection
    crs_data = crs_data

    # Create Figure -------------------------
    fig, axs = proplot.subplots(fig_array, 
                              aspect=10, axwidth=5, proj=crs_data,
                              hratios=tuple(np.ones(len(fig_array), dtype=int)),
                              includepanels=True, hspace=-2.3, wspace=0.15)

    for i in range(graphic_no):
        axs[i].format(lonlim=(24, 45), latlim=(34, 42),
                      labels=False, longrid=False, latgrid = False)


    # türkiye harici shapeler
    # Find the China boundary polygon.
    shpfilename = shpreader.natural_earth(resolution='10m',
                                                  category='cultural',
                                                  name='admin_0_countries')

    cts = ['Syria', 'Iraq', 'Iran', 'Azerbaijan', 'Armenia',
           'Russia', 'Georgia', 'Bulgaria', 'Greece', 'Cyprus',
           'Northern Cyprus', 'Turkey', 'Albania', 'North Macedonia',
           'Montenegro', 'Serbia', 'Italy']


    for country in shpreader.Reader(shpfilename).records():
        if country.attributes['ADMIN'] in cts:
            count_shp = country.geometry

            for i in range(graphic_no):
                axs[i].add_geometries([count_shp], cartopy.crs.PlateCarree(),
                                          facecolor='none', edgecolor = 'black',
                                          linewidth = 0.5, zorder = 2.2,)
    
    # graphic
    for i in range(graphic_no):
        mesh = axs[i].pcolormesh(data_df['longitude'], data_df['latitude'],
                             data_df[i], norm = norm,
                             cmap = cmap, vmin = vmin, vmax = vmax,
                             zorder = 2.1)

    # CMAP ----------------------

    cbar = fig.colorbar(mesh, ticks=ticks, loc='b', drawedges = False, shrink=1, space = -1.1, aspect = 30, )
    cbar.ax.tick_params(labelsize=11, )
    cbar.set_label(label='{} Anomaly ({}) | ERA5'.format(var_name, unit), size=16, loc = 'center', y=0.35, weight = 'bold')
    cbar.outline.set_linewidth(2)
    cbar.minorticks_off()
    cbar.ax.get_children()[4].set_color('black')
    cbar.solids.set_linewidth(1)


    # TEXT
    # Build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    
    title_method = data_df.dims[0]
    for i in range(graphic_no):
        
        if method=='Month':
            axs[i].set_title(r'{}: {}'.format(method, data_df[i].month.values), fontsize = 12,
                         loc = 'left', pad = -14, y = 0.01, x=0.020, weight = 'bold',)
        elif method=='Season':
            axs[i].set_title(r'{}: {}'.format(method, data_df[i].season.values), fontsize = 12,
                         loc = 'left', pad = -14, y = 0.01, x=0.020, weight = 'bold',)
        
    # savefig    
    plt.savefig(fr'pics/report/{method.lower()}_anom_{var_short}_graph.jpeg',
                bbox_inches='tight', optimize=False, progressive=True, dpi=300)