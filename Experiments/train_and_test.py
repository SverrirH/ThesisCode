import wandb
import wandb
import numpy as np
from sklearn.linear_model import *
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import *
from helpers import apply_scaling, apply_indexes, MinMaxScaler
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def train_and_test(model_constructor,
                   input_data,
                   output_data,
                   weight_func,
                   config,
                   standardizer=StandardScaler,
                   save_models = False,
                   **fit_kwargs):

    # Logging dicts
    models, input_output_pred = {}, {}

    # cross_validation
    cross_val = TimeSeriesSplit(5)
    index = np.arange(0,len(output_data))
    splits = cross_val.split(index)
    
    for i, (train, val) in enumerate(splits):
        fold_data = {}
        fold_data['fold'] = i

        # Apply indexes
        input_train = input_data[train]
        fold_data['Y_train'] = output_data[train]
        input_val = input_data[val]
        fold_data['Y_val'] = output_data[val]
        
        # Unfold data for standardiztaion
        n,f,p = input_data.shape
        input_train = input_train.reshape((-1,p))
        input_val = input_val.reshape((-1,p))
        
        # Apply standardization and re-fold for training
        std = standardizer()
        fold_data['X_train'] = std.fit_transform(input_train).reshape((-1,f,p))
        fold_data['X_val'] = std.transform(input_val).reshape((-1,f,p))
        
        # Train model and predict
        model = model_constructor()
        model.fit(fold_data,
                  weight_func=weight_func)
        
        fold_data['Y_train_pred'] = model.predict(fold_data['X_train'])
        fold_data['Y_val_pred'] = model.predict(fold_data['X_val'])
        
        input_output_pred[i] = fold_data
        
        if save_models:
            models[i] = model
        else:
            models[i] = []
            del model
            tf.keras.backend.clear_session()

        
    return {'models': models, 'iop': input_output_pred}

