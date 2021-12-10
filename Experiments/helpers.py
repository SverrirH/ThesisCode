import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import *
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import *
# from dataProcessingHelpers import sliding_window
from numpy.lib.stride_tricks import sliding_window_view
import os
import pandas as pd
from collections import defaultdict

def sliding_window(a, window):
    from numpy.lib.stride_tricks import as_strided
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return as_strided(a, shape=shape, strides=strides)

def make_name_string(name,args_dict):
    key_string = '_'.join([str(key[0]) + '_' + str(args_dict[key]) for key in args_dict])
    return name + '_' + key_string

def last_observations(x, prev_obs):
    n,*p = x.shape
    x_padded = np.concatenate([np.zeros((prev_obs,*p)),x])[:-1]
    windowed = np.lib.stride_tricks.sliding_window_view(x_padded, (prev_obs,2), )
    return windowed.squeeze()


class mean_error():
    def __init__(self, axis, agg_func, name):
        self.axis = axis
        self.agg_func = agg_func
        self.name = name

    def __call__(self, y_pred, y_true, weight_func = None):
        assert y_pred.shape == y_true.shape
        resids = np.array(y_pred) - np.array(y_true)
        resids_w_agg = self.agg_func(resids)
        
        if type(weight_func) != type(None):
            weights = weight_func(y_true)
            if resids_w_agg.ndim == 2:
                weights = weights[:,np.newaxis]
            if resids_w_agg.ndim == 3:
                weights = weights[:,np.newaxis,np.newaxis]
            resids_w_agg = resids_w_agg * weights
            
        error = np.mean(resids_w_agg, axis=self.axis)
        return error

class mean_squared_error(mean_error):
    def __init__(self, axis=None):
        super().__init__(axis, agg_func=lambda x: x**2, name='mean_squared_error')


class mean_absolute_error(mean_error):
    def __init__(self, axis=None):
        super().__init__(axis, agg_func=np.abs, name='mean_absolute_error')


def use_old_forecasts_as_observations(X, NWP_dates, n_obs):
    NWP_dates = np.array(NWP_dates)
    start_date = NWP_dates[0]
    end_date = NWP_dates[-1]
    dates = pd.date_range(start_date,end_date,freq='6h')
    df_forecasts = pd.DataFrame(index=dates,columns = range((1-n_obs),61))
    for col in df_forecasts.columns:
        if col > 0:
            df_forecasts.loc[NWP_dates,col] = [(index,col) for index in NWP_dates]
            
    # The age of the forecast being used to fill in old observations
    # if it's desireable to use older forecasts due to something like
    # convergence of the forecasted rainfall, then this is the parameter
    forecast_age = 1
    for i in range(np.ceil(n_obs / 6).astype(int)):
        i_start = 1-6*(i+1)
        i_end = -6*i
        row_shift = (i+forecast_age)
        col_shift = -(i+forecast_age)*6
        df_forecasts.loc[:,i_start:i_end] = df_forecasts.shift(row_shift,axis=0).shift(col_shift,axis=1).loc[:,i_start:i_end]
        

    date_to_index = defaultdict(lambda : [])
    date_to_index.update({date:index for index,date in enumerate(NWP_dates)})
    old_forecasts = np.zeros((X.shape[0],n_obs,X.shape[2]))
    X_with_obs = np.concatenate([old_forecasts,X],axis=1)

    for to_col in range(n_obs):
        for to_date,fro in df_forecasts.iloc[:,to_col].iteritems():
            if type(fro) == type(np.nan):
                continue
            (from_date,from_column) = fro
            from_row = date_to_index[from_date]
            from_col = from_column - 1
            to_row = date_to_index[to_date]
            X_with_obs[to_row,to_col] = X_with_obs[from_row,from_col+n_obs]
            
    return X_with_obs

class MinMaxScaler():
    '''
        test_array_a = np.arange(1000*100*10*5).reshape(1000,100,10,5)
        tmp = MinMaxScaler().fit(test_array_a.reshape(-1,100*10*5))
        b = tmp.transform(test_array_a.reshape(-1,100*10*5)).reshape(1000,100,10,5)
        tmp = MinMaxScaler().fit(test_array_a)
        a = tmp.transform(test_array_a)
        np.all(a == b)
        
        -------
        
        A = np.c_[X1_tmp, X2_tmp, X2_tmp]
        B = [X1_tmp, X2_tmp, X2_tmp]

        MMS1 = MinMaxScaler().fit(A)

        MMS2 = MinMaxScaler().fit(B[0])
        MMS3 = MinMaxScaler().fit(B[1])
        MMS4 = MinMaxScaler().fit(B[2])

        As = MMS1.transform(A)

        Bs1 = MMS2.transform(B[0])
        Bs2 = MMS3.transform(B[1])
        Bs3 = MMS3.transform(B[2])
        Bs = np.c_[Bs1,Bs2,Bs3]
        np.all(As == Bs)
        
        
        A_mo = MMS1.min_vals_obs
        B_mo = np.c_[MMS2.min_vals_obs,MMS3.min_vals_obs]
        print(np.all(A_mo == B_mo))

        A_mo = MMS1.max_vals_obs
        B_mo = np.c_[MMS2.max_vals_obs,MMS3.max_vals_obs]
        print(np.all(A_mo == B_mo))
    '''
    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val
        pass

    def fit(self, x):
        self.min_vals_obs = x.min(axis=0)[np.newaxis]
        self.max_vals_obs = x.max(axis=0)[np.newaxis]
        return self
    
    def __apply_all_but_last(self, x, func): 
        return np.apply_over_axes(func, x, range(x.ndim - 1))
    
    def transform(self, x):
        # For numerical stability
        eps = 1e-9
        # find the min and max-values of the input
        min_obs = self.__apply_all_but_last(x,np.min)
        max_obs = self.__apply_all_but_last(x,np.max)
        
        # Which variables to leave as is (mainly to handle the one-hot-encoding of some columns)
        already_to_scale = (min_obs == self.min_val) & (max_obs == self.max_val)
        
        # Scale current input with the same min/max values as the fit(x) data to between 0 and 1
        neutral_0_1 = (x - self.min_vals_obs) / \
            (self.max_vals_obs - self.min_vals_obs + eps)
            
        # scaling and offest defined by the __init__ paramters
        scale = self.max_val - self.min_val
        offset = self.min_val
        
        # Scale the 0 and 1 to the defined min and max
        scaled = offset + scale * neutral_0_1
        
        # Return the ones already in that form to the original (since this can otherwise cause erros)
        scaled[...,already_to_scale.ravel()] = x[...,already_to_scale.ravel()]
        return scaled

def find_experiment_directory():
    dirname = os.path.dirname(os.path.realpath(__file__) )
    return dirname

def rename_append(x): return (lambda a: str(a) + x)


def get_future_hour(x): return (
    x.index + datetime.timedelta(hours=int(x.name))).hour


def make_dataframe(input_data, previous_values, future_values, previous_or_future, current_only):
    input_vals = input_data.values.ravel()
    slide_1 = sliding_window(
        input_vals, previous_values + future_values).squeeze()
    slide_2 = sliding_window(slide_1, previous_values)

    slide_1_index = sliding_window(
        input_data.index.values.ravel(), previous_values + future_values).squeeze()
    slide_2_index = sliding_window(slide_1_index, previous_values)

    if current_only:
        values = np.asarray(slide_2[:, 0, -1])
        index = np.asarray(slide_2_index[:, 0, -1])
        columns = [0]
    if previous_or_future:
        values = np.asarray(slide_2[:, 0, :])
        index = np.asarray(slide_2_index[:, 0, -1])
        columns = range(-previous_values+1, 1)
    else:
        values = np.asarray(slide_2[:, :, -1])
        index = np.asarray(slide_2_index[:, 0, -1])
        columns = range(0, future_values+1)

    return pd.DataFrame(values, index=index, columns=columns)


def ohe_time_of_day_from_df(df):
    n,p = df.shape
    future_hours_raw = df.apply(get_future_hour).values.ravel()
    hour_ohe_seq = OneHotEncoder().fit_transform(future_hours_raw.reshape(-1, 1))
    hour_ohe_seq = hour_ohe_seq.toarray().reshape(-1, p, 24)
    return hour_ohe_seq


def sw_func(
    y,
    thresholds,
    rel_weigths,
    ):
    '''
    - n_samples x n_sensors
        return: (n_samples, ) weights
    - n_samples x n_steps x n_sensors
        return: (n_samples, n_steps)
    ''' 
    y_shape = y.shape

    # Specific handling for classification (VERY UNCERTAIN THAT THIS IS A CORRECT IMPLEMENTATION)
    # if classification:
    #     weights = np.ones(y.shape)
    #     for i, w in enumerate(rel_weigths):
    #         weights[y[..., 0, i] == 1, 0] = w
    #         weights[y[..., 1, i] == 1, 1] = w
    #     return weights.mean(axis=(2, 3))
    
    weights = np.ones(y.shape)
    for (t1, t2), w in zip(thresholds.T, rel_weigths):
        weights[y[..., 0] >= t1, 0] = w
        weights[y[..., 1] >= t2, 1] = w

    sample_weights = np.apply_over_axes(np.sum, weights, range(1,weights.ndim)).ravel()
    return sample_weights/np.min(sample_weights) # So the smallest weight is 1


def apply_indexes(input_data, train, val, xt_shape):
    if type(input_data) == list:
        X_train = []
        X_val = []
        for inp in input_data:
            X_train.append(xt_shape(inp[train]))
            X_val.append(xt_shape(inp[val]))
        return X_train,X_val
    else:
        X_train = xt_shape(input_data[train])
        X_val = xt_shape(input_data[val])
        return X_train,X_val
            
def apply_scaling(input_data,train,standardizer):
    if type(input_data) == list:
        return_list = []
        for inp in input_data:
            # Fit on training data
            std = standardizer().fit(inp[train])
            # apply to all data
            inp_tmp = std.transform(inp)
            return_list.append(inp_tmp)
        return return_list
    else:
        std = standardizer().fit(input_data[train])
        return_data = std.transform(input_data)
        return return_data
    
    
def check_missing_values(X1,X2,X3,Y1,Y2):
    print('Ratio of nan to all values')
    print((np.sum(np.isnan(X1))/np.product(X1.shape)).sum())
    print((np.sum(np.isnan(X2))/np.product(X2.shape)).sum())
    print(np.sum(np.isnan(X3))/np.product(X3.shape))
    print(np.sum(np.isnan(Y1))/np.product(Y1.shape))
    print(np.sum(np.isnan(Y2))/np.product(Y2.shape))

    print()
    print('Ratio of rows with nan values to all rows')
    print(np.any(np.isnan(X1), axis=1).sum()/X1.shape[0])
    print(np.any(np.isnan(X2), axis=1).sum()/X2.shape[0])
    print(np.any(np.isnan(X3), axis=1).sum()/X3.shape[0])
    print(np.any(np.isnan(Y1), axis=1).sum()/Y1.shape[0])
    print(np.any(np.isnan(Y2), axis=1).sum()/Y2.shape[0])
    
def create_dataset(series_list,lag):
    
    # Start by joining all data and recomputing with nan-values so timing isn't affected
    df_combined = pd.concat(series_list,axis=1)
    
    df_combined = df_combined.resample('1h').mean() 
    combined_index = df_combined.index
    date_sequences = sliding_window_view(combined_index, lag).squeeze()
    
    output_data = []
    for s in series_list:
        if isinstance(s,pd.Series):
            transformed = s.resample('1h').mean().loc[date_sequences.ravel()].values.reshape(date_sequences.shape)
            output_data.append(transformed)
        else:
            transformed = s.resample('1h').mean().loc[date_sequences.ravel()].values.reshape(date_sequences.shape + (s.shape[1],))
            output_data.append(transformed)
            
    return output_data,date_sequences

def get_rain_gauge_or_radar_data(threshold_dict,
                                 temperature_name,
                                 obs,
                                 d_start,d_end,
                                 URL_save_data = None,
                                 gauge_name = '1475_R_manual',
                                 return_just_dates = False,):
    GD_dirname = find_experiment_directory()
    if type(URL_save_data) != type(None):
        GD_dirname = URL_save_data
    drainage_names = np.array(list(threshold_dict.keys()))
    
    # Radar data
    df_radar = np.load(f'{GD_dirname}/Data/20x20cappi_rainfall.pkl',allow_pickle=True)
    df_radar = df_radar.resample('900s').mean().shift(4).resample('1h').mean() # helps a lot, since the mean value aggregates upwards but the drainage data is aggregated downwards
    
    # rain_gauge data
    df_gauge = pd.read_pickle(f'{GD_dirname}/Data/rain_gauges_combined.pkl')    
    
    # Temperature data (Added '_T' to differentiate between temperature and rain-gauge data)
    df_temperature = pd.read_pickle(f'{GD_dirname}/Data/temperature_data.pkl').rename(rename_append('_T'), axis=1)

    # Drainage flow data (Drop MOS-HOl and KOP-HAB since they are irrelevant)
    df_target = pd.read_pickle(f'{GD_dirname}/Data/drainage_Flow.pkl').drop(['MOS-HOL', 'KOP-HAB'], axis=1)

    Y_all = df_target.loc[d_start:d_end, drainage_names].copy()

    Xt_all = df_temperature.loc[d_start:d_end, temperature_name]#X.loc[d_start:d_end,temperature_name].copy()
    Xrg_all = df_gauge.loc[d_start:d_end,gauge_name].copy()
    Xra_all = df_radar.loc[d_start:d_end].copy()
    


    [Xt_all_seq_dirty,Y_all_seq_dirty,Xra_all_seq_dirty,Xrg_all_seq_dirty],dates_all_seq_dirty =\
        create_dataset([Xt_all,Y_all,Xra_all,Xrg_all],obs)
        
    # Because of the .squeeze() in the create dataset, this id done to standardize the shapes
    n = len(Xt_all_seq_dirty)
    Xt_all_seq_dirty = Xt_all_seq_dirty.reshape(n,-1,1)
    Y_all_seq_dirty = Y_all_seq_dirty.reshape(n,-1,2)
    Xra_all_seq_dirty = Xra_all_seq_dirty.reshape(n,-1,400)
    Xrg_all_seq_dirty = Xrg_all_seq_dirty.reshape(n,-1,1)
    
    # Take all but last value away from data where lagged value isn't necessary
    if type(Y_all_seq_dirty) == list:
        Y_all_seq_dirty = np.concatenate([i[:,:,np.newaxis] for i in Y_all_seq_dirty],axis=2)

    Y_all_seq_dirty = Y_all_seq_dirty[:,-1]

    # Drop rows with missing input data or all output data missing
    nan_rows = np.isnan(Xt_all_seq_dirty).any(axis=1).ravel() |\
        np.isnan(Xra_all_seq_dirty).any(axis=(1,2)).ravel() |\
            np.isnan(Xrg_all_seq_dirty).any(axis=1).ravel() |\
                np.isnan(Y_all_seq_dirty).any(axis=1).ravel()

    good_rows = ~nan_rows
    Xt_all_seq = Xt_all_seq_dirty[good_rows]
    Xra_all_seq = Xra_all_seq_dirty[good_rows]
    Xrg_all_seq = Xrg_all_seq_dirty[good_rows]

    Xt_all_single = Xt_all_seq_dirty[good_rows][:,-1]
    Xra_all_single = Xra_all_seq_dirty[good_rows][:,-1]
    Xrg_all_single = Xrg_all_seq_dirty[good_rows][:,-1]

    Y_all_single = Y_all_seq_dirty[good_rows]

    dates_all_seq = dates_all_seq_dirty[good_rows]

    hour_ohe_seq = OneHotEncoder().fit_transform(pd.DatetimeIndex(dates_all_seq.ravel()).hour.values.reshape(-1,1))
    hour_ohe_seq = hour_ohe_seq.toarray().reshape(-1,obs,24)
    hour_ohe_single = hour_ohe_seq[:,-1]

    X1a = Xra_all_seq
    X1b = Xrg_all_seq
    X2 = Xt_all_seq
    X3 = hour_ohe_seq
    Y12 = Y_all_single
    
    if return_just_dates:
        return dates_all_seq
    else:
        return X1a,X1b,X2,X3,Y12
    
def get_NWP_data(threshold_dict,temperature_name,max_lag,pred_dist,d_start,d_end, URL_save_data = './Data/', return_just_dates = False):
    GD_dirname = find_experiment_directory()
    
    drainage_names = np.array(list(threshold_dict.keys()))
    
    # Radar data
    df_NWP = np.load(f'{GD_dirname}/Data/df_NWP.pkl', allow_pickle=True)

    # Temperature data (Added '_T' to differentiate between temperature and rain-gauge data)
    df_temperature = pd.read_pickle(f'{GD_dirname}/Data/temperature_data.pkl').rename(rename_append('_T'), axis=1)

    # Drainage flow data (Drop MOS-HOl and KOP-HAB since they are irrelevant)
    df_target = pd.read_pickle(f'{GD_dirname}/Data/drainage_Flow.pkl').drop(['MOS-HOL', 'KOP-HAB'], axis=1)

    # Temperature forecast data
    df_spa = pd.read_pickle(f'{GD_dirname}/Data/temperature_forecast_data.pkl')

    df_Xto_past = make_dataframe(
        df_temperature.loc[:, temperature_name], max_lag, pred_dist, True, False)
    df_Xt_past_future = pd.concat(
        [df_Xto_past, df_spa], axis=1).rename(lambda x: int(x), axis=1)

    # Predicted 24 hour rolling temperature 67 hours into the future
    df_Xtr_future = df_Xt_past_future.loc[:, 1:pred_dist]
    
        
    df_X_NWP = df_NWP.loc[d_start:d_end].copy()

    # to remove the effect of differencing from last cumulative value
    df_X_NWP[df_X_NWP < 0] = 0
    df_X_temp = df_Xtr_future.loc[d_start:d_end]
    Y_tmp = df_target.loc[d_start:d_end, drainage_names].copy()
    df_target1_future = make_dataframe(
        Y_tmp.iloc[:, 0], max_lag, pred_dist, False, False).loc[:, 1:]
    df_target2_future = make_dataframe(
        Y_tmp.iloc[:, 1], max_lag, pred_dist, False, False).loc[:, 1:]


    i1 = set(df_X_NWP.dropna(axis=0).index)
    i2 = set(df_X_temp.dropna(axis=0).index)
    i3 = set(df_target2_future.dropna(axis=0).index)
    i4 = set(df_target1_future.dropna(axis=0).index)
    common_indexes = i1.intersection(i2).intersection(i3).intersection(i4)
    common_indexes = sorted(common_indexes)


    Y1 = df_target1_future.loc[common_indexes].values
    Y2 = df_target2_future.loc[common_indexes].values
    X1 = df_X_NWP.loc[common_indexes].values.reshape(-1, 66, 10, 10)[:, :pred_dist]
    X2 = df_X_temp.loc[common_indexes].copy().values  # X.iloc[:,10*10*66:].values
    X3 = ohe_time_of_day_from_df(df_target2_future.loc[common_indexes])

    Y1_tmp = Y1.reshape(Y1.shape + (1,))
    Y2_tmp = Y2.reshape(Y1.shape + (1,))
    Y12 = np.c_[Y1_tmp, Y2_tmp]
    
    if return_just_dates:
        return common_indexes
    else:
        
        return X1,X2,X3,Y12

def make_scheduler(config):
    initial_learning_rate = config['learning_rate']['inital_log'],
    final_learning_rate = config['learning_rate']['final_log'],
    intervals = config['learning_rate']['intervals'],
    epochs = config['epochs']

    learning_rates = np.logspace(initial_learning_rate,
                                final_learning_rate,
                                intervals)
    def lr_schedule(epoch, lr):
        return learning_rates[int(epoch/epochs * intervals)]

    return lr_schedule

def scheduler_callback(config):
    lr_schedule_function = make_scheduler(config)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule_function, verbose=0)
    return lr_scheduler

def early_stopper_callback(config):
    early_stopper = tf.keras.callbacks.EarlyStopping(**config['early_stopper'])
    return early_stopper

def construct_sequential(layer_list, seq_name):
    from tensorflow.keras import Sequential
    layers = []
    for name,params in layer_list.items():
        if 'Dense' in name:
            layers.append(Dense(**params))
    return Sequential(layers,name=seq_name)
        