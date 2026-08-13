"""
Author: Md Mahbub Alam
File: Shared training configuration for vessel trajectory prediction

The same area configuration is used for Non-PINN and PINN models.
"""


RANDOM_SEED = 42

RANDOM_SEEDS = {
    'python': RANDOM_SEED,
    'numpy': RANDOM_SEED,
    'tensorflow': RANDOM_SEED,
    'data_shuffle': RANDOM_SEED,
}

COMMON_HYPERPARAMETERS = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': 'mse',
    'metrics': ('mae', 'mse'),
    'batch_size': 32,
    'epochs': 50,
    'sampling_interval_minutes': 2,
    'input_timesteps': 15,
    'output_timesteps': 15,
    'input_window_minutes': 30,
    'prediction_horizon_minutes': 30,
}

AREA_CONFIGS = {
    'arctic': {
        'name': 'Arctic',
        'early_stopping': {
            'monitor': 'val_loss',
            'patience': 15,
            'restore_best_weights': True,
            'mode': 'min',
        },
        'reduce_lr_on_plateau': {
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 7,
            'min_lr': 1e-6,
            'mode': 'min',
        },
    },
    'georgia': {
        'name': 'Strait of Georgia',
        'early_stopping': {
            'monitor': 'val_loss',
            'patience': 10,
            'restore_best_weights': True,
            'mode': 'min',
        },
        'reduce_lr_on_plateau': {
            'monitor': 'val_loss',
            'factor': 0.3,
            'patience': 5,
            'min_lr': 1e-6,
            'mode': 'min',
        },
    },
}


__all__ = [
    'RANDOM_SEED',
    'RANDOM_SEEDS',
    'COMMON_HYPERPARAMETERS',
    'AREA_CONFIGS',
]
