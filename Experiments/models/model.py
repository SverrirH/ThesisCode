from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, Reshape, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow import random, losses
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow import device
import numpy as np
import wandb

from models.model_helpers import parallelize
from models.common_parameters import common_fit_params,random_seed

class model():
    def __init__(self, config):
        random.set_seed(random_seed)
        self.config = config
        
        self.model1 = self.construct_model()
        self.model2 = self.construct_model()
        self.model = parallelize(self.model1,self.model2)
        
        self.optimizer = Adam(learning_rate=self.config['lr'])
        self.model.compile(optimizer=self.optimizer,
                           loss=losses.MSE)
        

    def construct_model(self):
        filters = self.config['filters']
        units = self.config['units']
        regularization = self.config['regularization']
        obs = self.config['obs']
        p = self.get_input_shape()
        model = Sequential()
        model.add(Input((obs,p)))
        if filters > 0:
            model.add(Conv1D(filters = filters,
                             kernel_size = 1,
                             kernel_regularizer=l2(regularization)))
            regularization = 0
        if units > 0:
            model.add(LSTM(units = units,kernel_regularizer=l2(regularization)))
            regularization = 0
        model.add(Flatten())
        model.add(Dense(units = 1,activation='linear',kernel_regularizer=l2(regularization)))
        return model
        
            
    def get_input_shape(self):
        config = self.config
        if config['data_source'] == 'NWP':
            p = 125
        elif config['data_source'] == 'radar':
            p = 425
        elif config['data_source'] == 'gauge':
            p = 26
        return p
   
    def fit(self, fold_data, weight_func, debug=False):
        config = self.config
        
        X = fold_data['X_train']
        Y = fold_data['Y_train']
        
        X_val = fold_data['X_val']
        Y_val = fold_data['Y_val']

        fold_data['sample_weights'] = weight_func(Y)

        
        self.model.fit(X, Y,
                       validation_data=(X_val, Y_val),
                       sample_weight=fold_data['sample_weights'],
                       
                       batch_size = config['batch_size'],
                       
                       epochs = common_fit_params['epochs'],
                       verbose = common_fit_params['verbose'],
                       shuffle = common_fit_params['shuffle'],
                       # added wandb afterwards since it has to be called after init()
                       callbacks = common_fit_params['callbacks'] + [wandb.keras.WandbCallback()],
                       )
    
    def predict(self, X):
        return self.model.predict(X)
