import wandb
import datetime
import argparse
import numpy as np
from train_and_test import train_and_test
from helpers import get_NWP_data, get_rain_gauge_or_radar_data
from helpers import sw_func, find_experiment_directory, make_name_string
from helpers import use_old_forecasts_as_observations
from evaluator import only_MSE_and_GSS
import sys
from tensorflow import device

from models.model import model


# Fixed paramters
project = "Thesis Experiment 2 and 3"
# project = "Thesis Experiment 2 and 3 test"
d_start = datetime.datetime(2015, 1, 1)
d_end = datetime.datetime(2019, 12, 31, 23, 59, 59)
max_lag = 24
pred_dist = 60
t_roll = 24
threshold_dict = {
    "RVK-GEL": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "RVK-BOD": [0, 15,  30,  45,  60,  75,  90,  105, 120, 135, 150],
}
thresholds = np.array(list(threshold_dict.values()))
temperature_name = "1475_T"

experiment_dir = find_experiment_directory()
data_directory = f"{experiment_dir}/Data/"
save_dir = f"{experiment_dir}/predictions/"

early_stopper_params = {
    'monitor': 'val_loss',
    'min_delta': 0,
    'patience': 20,
    'verbose': 0,
    'mode': 'min',
    'baseline': None,
    'restore_best_weights': True,
    }

evaluation_config = {
    'eval_forecast_start': 24,
    'eval_forecast_end':30,
}


def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--regularization",type=float,default=0.1,metavar="regularization",)
    parser.add_argument("-u","--units",type=int,default=0,metavar="units",)
    parser.add_argument("-f","--filters",type=int,default=0,metavar="filters",)
    parser.add_argument("-l","--lr",type=float,default=0.1,metavar="lr",)
    parser.add_argument("-w","--weights",type=str,default='a',metavar="lr",)
    parser.add_argument("-b","--batch_size",type=int,default=256,metavar="batch_size",)
    parser.add_argument("-d","--data_source",type=str,default='gauge',metavar="data_source",)
    parser.add_argument("-o","--obs",type=int,default='3',metavar="obs",)
    args = parser.parse_args()
    
    config_model = {
        'regularization':args.regularization,
        'units':args.units,
        'filters':args.filters,
        
        'lr': args.lr,
        'batch_size': args.batch_size,
        'data_source': args.data_source,
        'obs': args.obs,
    }
        
    # Get data
    if args.data_source == "NWP":
        X1, X2, X3, Y = get_NWP_data(threshold_dict,temperature_name,config_model['obs'],pred_dist,d_start,d_end)
        NWP_dates = get_NWP_data(threshold_dict,temperature_name,config_model['obs'],pred_dist,d_start,d_end,return_just_dates=True)
        n,f,*_ = X1.shape
        X1 = X1.reshape((n,f,-1))
        X2 = X2.reshape((n,f,-1))
        X3 = X3.reshape((n,f,-1))
        # Transform such that the shape is (n*f,obs,-1)
        # and for 'observations' of rainfall, we instead use old forecasts
        X_before_obs = np.concatenate([X1,X2,X3],axis=-1)
        n,f,p = X_before_obs.shape
        X_after_obs = use_old_forecasts_as_observations(X_before_obs, NWP_dates, config_model['obs']-1)
        X = np.lib.stride_tricks.sliding_window_view(X_after_obs, (1,config_model['obs'],p)).reshape(-1,config_model['obs'],p)
        
    elif args.data_source == "radar":
        X1,_,X2,X3,Y = get_rain_gauge_or_radar_data(threshold_dict,temperature_name,config_model['obs'],d_start,d_end)
        X = np.concatenate([X1, X2, X3],axis=-1)
    elif args.data_source == "gauge":
        _,X1,X2,X3,Y = get_rain_gauge_or_radar_data(threshold_dict,temperature_name,config_model['obs'],d_start,d_end)
        X = np.concatenate([X1, X2, X3],axis=-1)
        
    X = X.astype('float32')
    Y = Y.reshape(-1,2).astype('float32')
    
    # Sample weights function
    if args.weights == "a":
        relative_weights = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        sample_weights_function = lambda x : sw_func(x, thresholds, relative_weights)
    elif args.weights == "b":
        relative_weights = [0.5, 1, 1.5, 2,4,6,8,10,12,15,20]
        sample_weights_function = lambda x : sw_func(x, thresholds, relative_weights)
    
    run = wandb.init(config=config_model, project=project)
    model_constructor = lambda : model(config_model)
    
       
    # class min_ridge_class():
    #   def __init__(self,**kwargs):
    #     self.model = Ridge(config_model['regularization'])
    #     pass
    #   def fit(self,fold_data, weight_func,**kwargs):
    #     X = fold_data['X_train']
    #     Y = fold_data['Y_train']
    #     fold_data['sample_weights'] = weight_func(Y)
    #     n,*_ = X.shape
    #     self.model.fit(X.reshape(n,-1),Y,sample_weight = fold_data['sample_weights'])

    #     # return y_pred
        
    #     # tod = X[:,0,2:]
    #     # rest = X[:,:,:2]
    #     # X_new = np.concatenate([rest.reshape(n,-1),tod],axis=-1)
    #     # print(X.shape,X_new.shape)
    #     # self.model.fit(X_new,Y,sample_weight = fold_data['sample_weights'])
    #   def predict(self,X,**kwargs):
    #     # print('predicting')
    #     n,*_ = X.shape
    #     y_pred = self.model.predict(X.reshape(n,-1))
    #     return y_pred
        
    #     # tod = X[:,0,2:]
    #     # rest = X[:,:,:2]
    #     # X_new = np.concatenate([rest.reshape(n,-1),tod],axis=-1)
    #     # print('predict:',X_new.shape,X.shape)
    #     # y_pred = self.model.predict(X_new)
    #     # print(y_pred.shape,X_new.shape)
    #     # return y_pred
          
    # if args.filters == 99:
    #     print('Doing Ridge')
    #     from sklearn.linear_model import Ridge
    #     model_constructor = lambda : min_ridge_class()
          
    # with device('cpu:0'):
    with device('cpu:0'):
        model_results = train_and_test(
            model_constructor=model_constructor,
            input_data=X,
            output_data=Y,
            weight_func=sample_weights_function,
            config=config_model,
        )
    
    predictions = {fold:{'Y_val_pred':model_results['iop'][fold]['Y_val_pred'].astype('float32'),
                         'Y_train_pred':model_results['iop'][fold]['Y_train_pred'].astype('float32')} for fold in range(5)}
    file_name = make_name_string('Predictions',config_model)
    np.save(save_dir + file_name,predictions, allow_pickle=True)
    
    minimum_eval_dict = only_MSE_and_GSS(model_results["iop"],sample_weights_function,evaluation_config)
    wandb.log(minimum_eval_dict['summary'])
    run.finish()
    
    # observations = {fold:{'Y_val':model_results['iop'][fold]['Y_val'].astype('float32'),
    #                      'Y_train':model_results['iop'][fold]['Y_train'].astype('float32')} for fold in range(5)}
    # file_name = make_name_string('Observations',config_model)
    # np.save(save_dir + file_name,observations, allow_pickle=True)
    
if __name__ == '__main__':
    train_model()

'''
method: grid
parameters:
  batch_size:
    values:
    - 254
  lr:
    values:
    - 0.1
  data_source:
    values:
    - NWP
    - radar
    - gauge
  filters:
    values:
    - 0
    - 1
    - 5
  obs:
    values:
    - 1
    - 12
    - 24
    - 48
  regularization:
    values:
    - 0
    - 0.1
  units:
    values:
    - 0
    - 1
    - 5
  weights:
    values:
    - a
    - b
program: train_and_test_model.py
'''


# for ridge regression
# TODO: NWP and 
'''
method: grid
parameters:
  batch_size:
    values:
    - 254
  lr:
    values:
    - 0.1
  data_source:
    values:
    - gauge
    - radar
    - NWP
  filters:
    values:
    - 99
  obs:
    values:
    - 1
    - 12
    - 24
    - 48
  regularization:
    values:
    - 0
    - 0.1
  units:
    values:
    - 0
  weights:
    values:
    - a
    - b
program: train_and_test_model.py
'''