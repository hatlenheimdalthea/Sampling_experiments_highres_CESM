# libraries
import os
from pathlib import Path
from collections import defaultdict
import scipy
import sys
import random
import numpy as np
import xarray as xr
import glob
import pandas as pd
import joblib
import sklearn
from skimage.filters import sobel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, max_error, mean_squared_error, mean_absolute_error, median_absolute_error
import keras
from keras import Sequential, regularizers
from keras.layers import Dense, BatchNormalization, Dropout
from statsmodels.nonparametric.smoothers_lowess import lowess
import gcsfs
fs = gcsfs.GCSFileSystem()

#===============================================
# Masks
#===============================================

def network_mask(topo_path,lsmask_path):
    """
    Creates network mask. This masks out regions in the NCEP land-sea mask 
    (https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html)
    to define the open ocean. 
    
    Regions removed include:
    - Coast : defined by sobel filter
    - Batymetry less than 100m
    - Arctic ocean : defined as North of 79N
    - Hudson Bay
    - caspian sea, black sea, mediterranean sea, baltic sea, Java sea, Red sea
    
    Parameters
    ----------
    topo_path : str
    
    lsmask_path : 
    
    Returns
    ----------
    data : Xarray dataset with masked out regions as written above.
    
    """
    
    ### topography
    topo_file_ext = topo_path.split('.')[-1] # getting if zarr or nc
    ds_topo = xr.open_dataset(topo_path, engine=topo_file_ext)
    ds_topo = ds_topo.roll(longitude=1800, roll_coords='longitude')
    ds_topo['longitude'] = np.arange(0.05, 360, .1)

    ### Loads grids
    # land-sea mask
    # land=0, sea=1
    
    lsmask_file_ext = topo_path.split('.')[-1] # getting if zarr or nc
    ds_lsmask = xr.open_dataset(lsmask_path, engine=lsmask_file_ext).sortby('latitude').squeeze().drop('time')
    data = ds_lsmask['mask'].where(ds_lsmask['mask']>0)

    #data[0,:,:].plot()

    ### Define Latitude and Longitude
    longitude = ds_lsmask['longitude']
    latitude = ds_lsmask['latitude']
    
    ### Remove coastal points, defined by sobel edge detection
    # coast = (sobel(ds_lsmask['mask'])>0)
    # data = data.where(coast==0)
    
    ### Remove show sea, less than 100m
    ### This picks out the Solomon islands and Somoa
    data = data.where(ds_topo['Height']<-100)
    
    ### remove arctic
    # data = data.where(~((latitude>79)))
    # data = data.where(~((latitude>67) & (latitude<80) & (longitude>20) & (longitude<180)))
    # data = data.where(~((latitude>67) & (latitude<80) & (longitude>-180+360) & (longitude<-100+360)))

    ### remove caspian sea, black sea, mediterranean sea, and baltic sea
    # data = data.where(~((latitude>24) & (latitude<70) & (longitude>14) & (longitude<70)))
    
    ### remove hudson bay
    # data = data.where(~((latitude>50) & (latitude<70) & (longitude>-100+360) & (longitude<-70+360)))
    # data = data.where(~((latitude>70) & (latitude<80) & (longitude>-130+360) & (longitude<-80+360)))
    
    ### Remove Red sea
    # data = data.where(~((latitude>10) & (latitude<25) & (longitude>10) & (longitude<45)))
    # data = data.where(~((latitude>20) & (latitude<50) & (longitude>0) & (longitude<20)))
    
    ### remove Okhtosk
    # data = data.where(~((latitude>50) & (latitude<66) & (longitude>136) & (longitude<159)))

    data = data.roll(longitude=1800, roll_coords='longitude')
    data['longitude'] = np.arange(-179.95,180.05,.1)
    
    return data

#===============================================
# Data prep functions
#===============================================

def make_dates(dates_range_start, dates_range_end):
    
    """
    Creates string starting with desired start date for data, and ending with desired end date for data. 
    Makes pandas DatetimeIndex, made using pd.date_range functionality.
    
    Example input:
    
        # Define date range
        date_range_start = '1982-01-01T00:00:00.000000000'
        date_range_end = '2017-01-01T00:00:00.000000000'

        # create date string
        date_str = pre_saildrone.make_date_str(dates)
        
    Parameters
    ----------
    dates_range_start : 
    
    dates_range_end :
    
    Returns
    ----------
    dates_range : pandas DatetimeIndex object
        Range of dates for data to be retrieved between.
    dates_str : str
        String made up of start and end of date range for proper naming of files and file retrieval.
        
    Bugs
    ----------
    TAKES ONLY ONE TYPE OF DATE FORMAT TO CREATE THE STRING! CHANGE THAT!
    
    """
    dates = pd.date_range(start=date_range_start, 
                      end=date_range_end,freq='MS') + np.timedelta64(14, 'D')
    
    filename_date_start = dates_range_start.split('-')[0] + dates_range_start.split('-')[1]
    filename_date_end = dates_range_end.split('-')[0] + date_ranges_end.split('-')[1]
    dates_str = f'{filename_date_start}-{filename_date_end}'
    
    return dates, dates_str

def detrend_time(array_1d, N_time):
    """
    EXPLANATION OF FXN. Assumes 2d array can be filled in column-wise (i.e. time was the first dimension that generated the 1d array).
    
    Parameters
    ----------
    array_1d : np.array
        1d array
    N_time : int
        Length that the time dimension should be
        
    Returns
    ----------
    array_detrend_2d.flatten(order = 'C'): 1d array of original data less linear trend over time; 
        any location with at least one nan is returned as nan for all times
    
    """
    array_2d = array_1d.reshape(N_time,-1,order='C')
    nan_mask = (np.any(np.isnan(array_2d), axis=0))
    X = np.arange( N_time )
    regressions = np.polyfit(X, array_2d[:,~nan_mask], 1)
    lin_fit = (np.expand_dims(X,1) * regressions[0:1,:] + regressions[1:,:])
    array_detrend_2d = np.empty(shape=array_2d.shape)
    array_detrend_2d[:] = np.nan
    array_detrend_2d[:,~nan_mask] = array_2d[:,~nan_mask] - lin_fit
    
    return array_detrend_2d.flatten(order='C')

def calc_anom(array_1d, N_time, N_batch, array_mask0=None):
    
    """
    EXPLANATION OF FXN. Assumes 2d array can be filled in C order (i.e. time was the first dimension that generated the 1d array).
    Note: can include an extra array to use to adjust for values that should be set to 0 (refers to array_mask0 parameter).
    
    Parameters
    ----------
    array_1d : np.array
        1d array
    N_time : int
        Length that the time dimension should be
    N_batch : int
         Window for averaging ???
    
    array_mask0 : boolean
        Extra array to use for adjusting values that should be set to 0. Defaults to None.
        
    Returns
    ----------
    output : 
            
    """
    array_2d = array_1d.copy()
    if array_mask0 is not None:
        nan_mask = np.isnan(array_2d)
        mask0 = np.nan_to_num(array_mask0, nan=-1.0) <= 0
        array_2d[mask0] = np.nan
    array_2d = array_2d.reshape(N_time,-1,order='C')

    for i in range(-(-N_time//N_batch)):
        avg_val = np.nanmean(array_2d[(i*N_batch):((i+1)*N_batch),:])
        array_2d[(i*N_batch):((i+1)*N_batch),:] = array_2d[(i*N_batch):((i+1)*N_batch),:] - avg_val
    
    output = array_2d.flatten(order='C')
    if array_mask0 is not None:
        output[~nan_mask & mask0] = 0
    
    return output

#===============================================
# Calculate anoms from a mean seasonal cycle:
#===============================================

def calc_interannual_anom(df):
    
    """
    EXPLANATION OF FXN
    
    Parameters
    ----------
    df : 
        
    Returns
    ----------

    df_anom : 
    
    """
    
    # chl, sst, sss, xco2 all may have seasonal cycles in them
    DS = df.to_xarray() # get from multi-index back to an xarray for calculating mean seasonal cycle:
    DS_cycle = DS.groupby("time.month").mean("time")
    # Now get anomalies from this mean seasonal cycle:
    DS_anom = DS.groupby("time.month") - DS_cycle
    
    # print(DS_anom)
    
    DS2 = xr.Dataset(
        {
        'anom':(['latitude','longitude', 'time'], DS_anom.data                    
        )},

        coords={
        'latitude': (['latitude'],DS.latitude.data),
        'longitude': (['longitude'],DS.longitude.data),
        'time': (['time'],DS.time.data)
        })
        
    df_anom = DS2.to_dataframe()

    return df_anom

#==============================================================================================

def log_or_0(array_1d):
  
    """
    EXPLANATION OF FXN
    
    Parameters
    ----------
    1d array : np.array
        1d array
        
    Returns
    ----------
    output: log of 1d array or 0 for values <= 0
    
    """
    output_ma, output = array_1d.copy(), array_1d.copy()
    output_ma = np.ma.masked_array(output_ma, np.isnan(output_ma))
    output_ma = np.ma.log10(output_ma)
    output[~output_ma.mask] = output_ma[~output_ma.mask]
    output[output_ma.mask] = np.maximum(output[output_ma.mask],0)
    return output

def detrend_pco2(ensemble_dir_head, ens, member, start_date, end_date):
    
    """
    EXPLANATION OF FXN
    
    Parameters
    ----------
    ensemble_dir_head : str
        Path to head directory
    ens : str
        Ensemble name
    member : str
        Member name (an integer value) on which you are running
    start_date :
        *include notes on date format required*
    end_date: 
        *include notes on date format required*
    
    """
    
    member_dir = f"{ensemble_dir_head}/{ens}/member_{member}"
    xpco2_path = '/data/artemis/workspace/vbennington/NOAA_ESRL/atmos_pco2_3D_mon_198201-201701.nc'
    
    if ens == "CanESM2":
        # CanESM2 files are mislabeled as going to 201712
        pco2_path = f"{member_dir}/pCO2_2D_mon_{ens}{member}_1x1_198201-201712.nc"
    else:
        pco2_path = f"{member_dir}/pCO2_2D_mon_{ens}{member}_1x1_198201-201701.nc"
        
    pco2_subtract = xr.open_dataset(xpco2_path).pco2_subtract
    pco2_model = xr.open_dataset(pco2_path).pCO2
    latitude = pco2_model.latitude
    longitude = pco2_model.longitude
    time = pco2_model.time
    
    pco2_detrend = pco2_model
    
    pco2_detrend = pco2_model - np.array(pco2_subtract)
                 
    fname_out = f'/data/artemis/workspace/vbennington/detrend_atmos/LET/{ens}/detrended_pCO2_2D_mon_{ens}{member}_1x1_198201-201701.nc'
    ds3d_out = xr.Dataset(
        {
        'pco2_detrend':(['time','ylat','xlon'], pco2_detrend                       
        )},

        coords={
        'time': (['time'],pco2_model.time.data),
        'longitude': (['longitude'],pco2_model.longitude.data),
        'latitude': (['latitude'],pco2_model.latitude.data)
        })
        
    # Save to netcdf
    ds3d_out.to_netcdf(fname_out)
    
def smoothTriangle(data, degree):
    
    """
    EXPLANATION OF FXN
    
    Parameters
    ----------
    data : 
    
    degree : 
    
    Returns
    ----------
    smoothed : 
    
    """

    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

#===============================================
# Loading in data and creating features
#===============================================

def import_model_data(model_dir,
                       xco2_path, socat_path,files_ext='zarr'):
    
    """
    EXPLANATION OF FXN
    
    Parameters
    ----------
    ensemble_dir_head : str
    
    ens : str
    
    member : str
    
    dates_str : str
        String made up of user-specified start and end dates to look for files (salinity, chlorophyll, etc) corresponding to those dates.
        
    files_ext : str
        String of either 'nc' or 'zarr' for proper file opening. Defaults to 'zarr'.
    
    xco2_path : str
    
    sail_path : str
    
    Returns
    ----------
    DS : 
    inputs['xco2'] : 
    """

    # model_data_path = fs.glob(f"{model_dir}/regridded*.zarr")[0]
    # chl_clim_path = fs.glob(f"{model_dir}/chlclim*.zarr")[0]

    # model_data = xr.open_mfdataset('gs://'+member_path, engine=file_engine)
    # socat_mask_data = xr.open_mfdataset('gs://'+socat_path, engine=file_engine)
    # tmp = xr.open_mfdataset('gs://'+chl_clim_path, engine=file_engine).chl_clim

    #model_data_path = glob.glob(f"{model_dir}/regridded*.zarr")[0]
    #chl_clim_path = glob.glob(f"{model_dir}/chlclim*.zarr")[0]
    
    #model_data = xr.open_mfdataset(model_data_path, engine='zarr')
    #socat_mask_data = xr.open_mfdataset(socat_path, engine='zarr')
    #tmp = xr.open_mfdataset(chl_clim_path, engine='zarr').chl_clim

    CESM_path = xr.open_dataset('gs://leap-persistent/abbysh/ncar_files/processed/pco2_components_202002-202201.zarr', engine='zarr')
    socat_path = xr.open_dataset(socat_path, engine='zarr')
    xco2_path = xr.open_dataset(xco2_path, engine='zarr')
    
    inputs = {}

    times = CESM_path.time
    #times = CESM_path.SALT.sel(time=slice("2020-02-01","2022-01-01"))
    
    inputs['socat_mask'] = socat_path.socat_mask
    inputs['sss'] = CESM_path.SALT#.sel(time=slice("2020-02-01","2022-01-01"))
    inputs['sst'] = CESM_path.insitu_temp#.sel(time=slice("2020-02-01","2022-01-01"))
    inputs['chl'] = CESM_path.combined_chl#.sel(time=slice("2020-02-01","2022-01-01"))
    inputs['mld'] = CESM_path.HMXL#.sel(time=slice("2020-02-01","2022-01-01"))
    inputs['pCO2_residual'] = CESM_path.pco2_residual#.sel(time=slice("2020-02-01","2022-01-01"))
    inputs['pCO2'] = CESM_path.pCO2SURF#.sel(time=slice("2020-02-01","2022-01-01"))
    inputs['xco2'] = xco2_path.xco2#.sel(time=slice("2020-02-01","2022-01-01"))

    #Create Chl Clim 1970-1997 and then 1998-2017 time varying CHL:

    # larger number is total number of months
    # smaller number should be months from start until 1998
    # tmp2 = model_data.chl
    
    # chl_sat = np.empty(shape=(636,180,360))
    
    # for yr in range(1970,1998):
    #     chl_sat[(yr-1970)*12:(yr-1969)*12,:,:]=tmp

    # # print(tmp2[119:])
    
    # chl_sat[348:636,:,:]=tmp2[348:636,:,:]
    
    # chl2 = xr.Dataset({'chl_sat':(["time","ylat","xlon"],chl_sat.data)},
    #                 coords={'time': (['time'],tmp2.time.data),
    #                 'ylat': (['ylat'],tmp2.ylat.data),
    #                 'xlon':(['xlon'],tmp2.xlon.data)})
    # inputs['chl_sat'] = chl2.chl_sat

    for i in inputs:        
        if i != 'xco2':
            # inputs[i] = inputs[i].transpose('time', 'xlon', 'ylat')
            time_len = len(times)
            inputs[i].assign_coords(time=times[0:time_len])

    DS = xr.merge([inputs['sss'], inputs['sst'], inputs['mld'], inputs['chl'], inputs['pCO2_residual'], inputs['pCO2'], inputs['socat_mask'],
                   ], compat='override', join='override')
    
    return DS, inputs['xco2']

def create_features(df, N_time=492, N_batch = 12):
    
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    
    Returns
    ----------
    
    """
    
    df['mld_log'] = log_or_0(df['HMXL'].values)
    df['mld_anom'] = calc_anom(df['mld_log'].values, N_time, N_batch,array_mask0=df['HMXL'].values)
    df['mld_log_anom'] = calc_interannual_anom(df['mld_log'])
    
    df_mld = df.loc[(df['HMXL']>0),'HMXL']
    mld_grouped = df_mld.groupby(by=[df_mld.index.get_level_values('time').month, 'longitude','latitude']).mean()
    df = df.join(mld_grouped, on = [df.index.get_level_values('time').month, 'longitude','latitude'], rsuffix="_clim")
    df['mld_clim_log'] = log_or_0(df['HMXL_clim'].values)

    df['chl_log'] = log_or_0(df['combined_chl'].values)
    df['chl_log_anom'] = calc_interannual_anom(df['chl_log'])
    #df['chl_sat_log'] = log_or_0(df['chl_sat'].values)
    #df['chl_sat_anom'] = calc_interannual_anom(df['chl_sat_log'])
    
    df.rename(columns={'SALT':'sss'}, inplace=True)
    df['sss_anom'] = calc_anom(df['sss'].values, N_time, N_batch)

    df.rename(columns={'insitu_temp':'sst'}, inplace=True)
    df['sst_anom'] = calc_interannual_anom(df['sst'])
    
    days_idx = df.index.get_level_values('time').dayofyear
    lon_rad = np.radians(df.index.get_level_values('longitude').to_numpy())
    lat_rad = np.radians(df.index.get_level_values('latitude').to_numpy())
    df['T0'], df['T1'] = [np.cos(days_idx * 2 * np.pi / 365), np.sin(days_idx * 2 * np.pi / 365)]
    df['A'], df['B'], df['C'] = [np.sin(lat_rad), np.sin(lon_rad)*np.cos(lat_rad), -np.cos(lon_rad)*np.cos(lat_rad)]
    return df

def create_inputs(ensemble_dir_head, dates, N_time,
                  xco2_path, socat_path, topo_path, lsmask_path,
                  N_batch = 12):
   
    DS, DS_xco2 = import_model_data(ensemble_dir_head,
                                     xco2_path=xco2_path,socat_path=socat_path)

    # print(DS)
    df = DS.to_dataframe()
    df = create_features(df, N_time = N_time, N_batch = N_batch)

    ### HACK TO FIX NET MASK ###
    # net_mask = np.repeat(network_mask(topo_path,lsmask_path).transpose('longitude','latitude').to_dataframe()['mask'].to_numpy(),len(dates))
    df_array = df.to_xarray()
    netmask = network_mask(topo_path,lsmask_path)
    netmask_zeros = np.zeros((1800,3600,N_time))
    for month in range(N_time):
        netmask_zeros[:,:,month] = netmask.values

    new_netmask= xr.DataArray(
    data=netmask_zeros,
    dims=['latitude','longitude', "time"],
    coords=dict(
        longitude=df_array.longitude,
        latitude=df_array.latitude,
        time=df_array.time
    ))

    df_array['net_mask'] = new_netmask

    df_as_dataframe = df_array.to_dataframe()
    
    df = df_as_dataframe
    ### END OF HACK ###
    
    #df['xco2'] = np.repeat(DS_xco2.sel(time=slice("2020-02-01","2022-01-01")).values,1800*3600)
    #df['xco2'] = np.repeat(DS_xco2.sel(time=slice('2021-01-01','2021-12-16')).values,1800*3600)

    ### HACK TO FIX XCO2 ###
    xco2_values = DS_xco2.sel(time=slice("2020-02-01","2022-02-01")).values.flatten()
    df['xco2'] = np.tile(xco2_values, int(np.ceil(len(df) / len(xco2_values))))[:len(df)]
    ### END OF HACK ###
    
    return df


#===============================================
# Evaluation functions
#===============================================

def centered_rmse(y,pred):
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    y : 
    
    pred : 
    
    Returns
    ----------
    np.sqrt(np.square((pred - pred_mean) - (y - y_mean)).sum()/pred.size) : 
    
    """
    y_mean = np.mean(y)
    pred_mean = np.mean(pred)
    return np.sqrt(np.square((pred - pred_mean) - (y - y_mean)).sum()/pred.size)

def evaluate_test(y, pred):
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    y : 
    
    pred : 
    
    Returns
    ----------
    scores : 
    
    """
    scores = {
        'mse':mean_squared_error(y, pred),
        'mae':mean_absolute_error(y, pred),
        'medae':median_absolute_error(y, pred),
        'max_error':max_error(y, pred),
        'bias':pred.mean() - y.mean(),
        'r2':r2_score(y, pred),
        'corr':np.corrcoef(y,pred)[0,1],
        'cent_rmse':centered_rmse(y,pred),
        'stdev' :np.std(pred),
        'amp_ratio':(np.max(pred)-np.min(pred))/(np.max(y)-np.min(y)), # added when doing temporal decomposition
        'stdev_ref':np.std(y),
        'range_ref':np.max(y)-np.min(y),
        'iqr_ref':np.subtract(*np.percentile(y, [75, 25]))
        }
    return scores

#===============================================
# Train test split functions
#===============================================

def train_val_test_split(N, test_prop, val_prop, random_seeds, ens_count):
    
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    N : 
    
    test_prop : 
    
    val_prop : 
    
    random_seeds : 
    
    ens_count : 
    
    Returns
    ----------
    intermediate_idx : 
    train_val_idx : 
    train_idx : 
    val_idx : 
    test_idx : 
    
    """
    
    intermediate_idx, test_idx = train_test_split(range(N), test_size=test_prop, random_state=random_seeds[0,ens_count])
    train_idx, val_idx = train_test_split(intermediate_idx, test_size=val_prop/(1-test_prop), random_state=random_seeds[1,ens_count])
    return intermediate_idx, train_idx, val_idx, test_idx

def apply_splits(X, y, train_val_idx, train_idx, val_idx, test_idx):
    
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    X :
    
    y :
    
    train_val_idx : 
    
    train_idx : 
    
    val_idx : 
    
    test_idx : 
    
    Returns
    ----------
    X_train_val : 
    X_train : 
    X_val : 
    X_test :
    y_train_val : 
    y_train :
    y_val :
    y_test : 
    
    """
    
    X_train_val = X[train_val_idx,:]
    X_train = X[train_idx,:]
    X_val = X[val_idx,:]
    X_test = X[test_idx,:]

    y_train_val = y[train_val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    return X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test

# def cross_val_splits(train_val_idx, random_seeds, row, ens_count, folds=3):
#     """Didn't actually use this"""
#     idx = train_val_idx.copy()
#     np.random.seed(random_seeds[row,ens_count])
#     np.random.shuffle(idx)
#     list_val = np.array_split(idx, folds)
#     list_train = []
#     for i in range(folds):
#         list_train.append( np.concatenate(list_val[:i] + list_val[(i+1):]) )
#     return zip(list_train, list_val)

#===============================================
# NN functions
#===============================================

def build_nn(num_features, neurons=[512,256], act='relu', use_drop=True, drop_rate=0.5, learning_rate=0.01, reg=0.001):
    
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    num_features : int
        ??. Number of features.
    neurons : 
        ??. Defaults to [512,256]
    act : str
        ??. Defaults to 'relu'.
    use_drop : bool
        ??. Defaults to True.
    drop_rate : float
        ??. Defaults to 0.5.
    learning_rate : float
        ??. Defaults to 0.01.
    reg : float
        ??. Defaults to 0.001.
        
    Returns
    ----------
    model : 
    
    """
    
    model = Sequential()
    model.add(BatchNormalization(input_shape=(num_features,)))
    for i in range(len(neurons)):
        model.add(Dense(units=neurons[i], activation=act, kernel_regularizer=regularizers.l2(reg)))
        if use_drop:
            model.add(Dropout(drop_rate))
    model.add(Dense(units=1))

    model.compile(keras.optimizers.Adam(lr=learning_rate), loss='mse', metrics=['mse'])

    return model

def build_nn_vf(num_features, act='relu', learning_rate=0.01, reg=0.001):
    
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    num_features : 
    
    act : str
        ??. Defaults to 'relu'.
    learning_rate : float
        ??. Defaults to 0.01.
    reg : float
            ??. Defaults to 0.001.
    
    Returns
    ----------
    model : 
    
    """
    
    model = Sequential()
    model.add(BatchNormalization(input_shape=(num_features,)))
    model.add(Dense(units=500, activation=act, kernel_regularizer=regularizers.l2(reg)))
    model.add(Dense(units=500, activation=act, kernel_regularizer=regularizers.l2(reg)))
    model.add(Dense(units=1))

    model.compile(keras.optimizers.Adam(lr=learning_rate), loss='mse', metrics=['mse'])

    return model


#===============================================
# Saving functions
#===============================================

def save_clean_data(df, data_output_dir, member, dates):
    
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    df : 
    
    data_output_dir: 
    
    Returns
    ----------
    
    """
    
    print("Starting data saving process")

    init_date = str(dates[0].year) + format(dates[0].month,'02d')
    fin_date = str(dates[-1].year) + format(dates[-1].month,'02d')
    
    output_dir = f"{data_output_dir}/{ens}/{member}"
    fname = f"{output_dir}/MLinput_{ens}_{member.split('_')[-1]}_mon_01x01_{init_date}_{fin_date}.pkl"
    df.to_pickle(fname)
    print(f"{member} save complete")

def save_model(model, dates, model_output_dir, approach, run=None):
    
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    
    Returns
    ----------
    
    """
    
    print("Starting model saving process")

    init_date = str(dates[0].year) + format(dates[0].month,'02d')
    fin_date = str(dates[-1].year) + format(dates[-1].month,'02d')
    
    model_dir = f"{model_output_dir}"
    model_fname = f"{model_dir}/mlmodel_fC02_2D_{approach}_mon_01x01_{init_date}_{fin_date}.json"
    # print(model_fname) 
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save_model(model_fname)
    print("Save complete")

def save_recon(DS_recon, dates, recon_output_dir, approach, run=None):
    
    """
    EXPLANATION OF FXN.
    
    Parameters
    ----------
    
    Returns
    ----------
    
    """
    
    print("Starting reconstruction saving process")

    init_date = str(dates[0].year) + format(dates[0].month,'02d')
    fin_date = str(dates[-1].year) + format(dates[-1].month,'02d')
    
    recon_dir = f"{recon_output_dir}"
    # Path(recon_dir).mkdir(parents=True, exist_ok=True)
    recon_fname = f"{recon_dir}/recon_fC02residual_{approach}_mon_01x01_{init_date}_{fin_date}.zarr"

    print(recon_fname)
    DS_recon.to_zarr(f'{recon_fname}', mode='w')
    print("Save complete")
