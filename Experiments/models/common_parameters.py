from tensorflow.keras.callbacks import EarlyStopping
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
early_stopper_params = {
    'monitor': 'val_loss',
    'min_delta': 0,
    'patience': 20,
    'verbose': 0,
    'mode': 'min',
    'baseline': None,
    'restore_best_weights': True,
    }

early_stopper = EarlyStopping(**early_stopper_params)
random_seed = 0
reduce_learning_rate_callback = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=5,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=3,
    min_lr=0,
)

common_fit_params = {
    'epochs': 100,
    'verbose': 1,
    'shuffle': True,
    'callbacks': [early_stopper,
                  reduce_learning_rate_callback]
}