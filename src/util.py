import cartopy
import pyproj
import xarray as xr

model_field_map = {'t2m': 'T_2M', 'prec': 'TOT_PREC',
                   't': 'T', 'relhum': 'RELHUM',
                   'geopotential': 'FI'}

era_field_map = {'t2m': 't2m', 'prec': 'tp',
                 't': 't'}


class ModelProj():
    
    def __init__(self):
        
        self.pole_longitude = -147.0
        self.pole_latitude = 50.0
        
        # crs info for the model
        self.crs_info_model = cartopy.crs.RotatedPole(pole_longitude=self.pole_longitude, 
                                                     pole_latitude=self.pole_latitude, 
                                                     central_rotated_longitude=0).proj4_params

def save_verif_file(da, var, prepath, verif_type):
    da.name = var
    file_name = fr'model_{var}_{verif_type}_verif.nc'
    da.to_netcdf(prepath + file_name)
    return file_name + " created"
        
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
        
    else:
        raise Exception("Sorry, the only aggregation procedures supported are sum and mean")
        
    return da_aggr

def get_model_era_files(var_type, var, start_date, end_date):
    """
    Get the ERA5 and model files
    
    return da_model, da_era
    
    """
    
    era_path = fr"data/era5/era5_{var_type}_hourly_project_data_climate_modeling.nc"
    model_path = fr"data/model/model_{var_type}_6h.nc"
    
    var_era = era_field_map[var]
    var_model = model_field_map[var]
    
    
    da_era = xr.open_dataset(era_path)[var_era].sel(time=slice(start_date, end_date))
    da_model = xr.open_dataset(model_path)[var_model].sel(time=slice(start_date, end_date))
    
    #da_era = da_era.sel(time=slice(start_date, end_date)).resample(time=aggregate)
    #da_model = da_model.sel(time=slice(start_date, end_date)).resample(time=aggregate)
    
    #if var=='prec':
    #    da_era = da_era.sum()
    #    da_model = da_model.sum()
        
    #    print("Warning: aggregating by monthly sum")
        
        
    #else:
    #    da_era = da_era.mean()
    #    da_model = da_model.mean()
        
    #    print("Warning: aggregating by monthly mean")
        
    return da_model, da_era

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
        
    
    
    
    