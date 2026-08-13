"""
Author: Md Mahbub Alam
File: Vessel trajectory prediction models

Available models: LSTM, GRU, CNN, ConvLSTM, and TCN.
"""

from functools import partial
from numbers import Integral

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    ConvLSTM2D,
    Dense,
    Flatten,
    GlobalAveragePooling1D,
    GRU,
    Input,
    LSTM,
    Reshape,
    SpatialDropout1D,
)


AVAILABLE_MODELS = ('lstm', 'gru', 'cnn', 'convlstm', 'tcn')
MODEL_COMPLEXITY_LEVEL = ('simple', 'complex')


def _validate_model_shape(
        input_timesteps, input_features, output_timesteps, output_features):
    values = {
        'input_timesteps': input_timesteps,
        'input_features': input_features,
        'output_timesteps': output_timesteps,
        'output_features': output_features,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f'{name} must be a positive integer.')


def _validate_model_complexity_level(model_complexity_level):
    if not isinstance(model_complexity_level, str):
        raise TypeError('model_complexity_level must be a string.')
    model_complexity_level = model_complexity_level.lower().strip()
    if model_complexity_level not in MODEL_COMPLEXITY_LEVEL:
        raise ValueError(
            f"Unknown model complexity level '{model_complexity_level}'. "
            f'Choose from {MODEL_COMPLEXITY_LEVEL}.'
        )
    return model_complexity_level


def build_lstm_model(
        input_timesteps, input_features, output_timesteps, output_features,
        model_complexity_level='simple'):
    """Build the simple or complex LSTM trajectory model."""
    _validate_model_shape(
        input_timesteps, input_features, output_timesteps, output_features
    )
    model_complexity_level = _validate_model_complexity_level(
        model_complexity_level
    )

    if model_complexity_level == 'simple':
        recurrent_layers = [
            LSTM(64, return_sequences=False),
        ]
    else:
        recurrent_layers = [
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
        ]

    return Sequential([
        Input(shape=(input_timesteps, input_features)),
        *recurrent_layers,
        Dense(output_timesteps * output_features, activation='linear'),
        Reshape((output_timesteps, output_features)),
    ])


def build_gru_model(
        input_timesteps, input_features, output_timesteps, output_features,
        model_complexity_level='simple'):
    """Build the simple or complex GRU trajectory model."""
    _validate_model_shape(
        input_timesteps, input_features, output_timesteps, output_features
    )
    model_complexity_level = _validate_model_complexity_level(
        model_complexity_level
    )

    if model_complexity_level == 'simple':
        recurrent_layers = [
            GRU(64, return_sequences=False),
        ]
    else:
        recurrent_layers = [
            GRU(64, return_sequences=True),
            GRU(32, return_sequences=False),
        ]

    return Sequential([
        Input(shape=(input_timesteps, input_features)),
        *recurrent_layers,
        Dense(output_timesteps * output_features, activation='linear'),
        Reshape((output_timesteps, output_features)),
    ])


def build_cnn_model(
        input_timesteps, input_features, output_timesteps, output_features,
        model_complexity_level='simple'):
    """Build the simple or complex CNN trajectory model."""
    _validate_model_shape(
        input_timesteps, input_features, output_timesteps, output_features
    )
    model_complexity_level = _validate_model_complexity_level(
        model_complexity_level
    )

    minimum_timesteps = 3 if model_complexity_level == 'simple' else 5
    if input_timesteps < minimum_timesteps:
        raise ValueError(
            f'{model_complexity_level} CNN requires at least '
            f'{minimum_timesteps} input timesteps.'
        )

    convolution_layers = [
        Conv1D(filters=64, kernel_size=3, activation='relu'),
    ]
    if model_complexity_level == 'complex':
        convolution_layers.append(
            Conv1D(filters=32, kernel_size=3, activation='relu')
        )

    return Sequential([
        Input(shape=(input_timesteps, input_features)),
        *convolution_layers,
        GlobalAveragePooling1D(),
        Dense(output_timesteps * output_features, activation='linear'),
        Reshape((output_timesteps, output_features)),
    ])


def build_convlstm_model(
        input_timesteps, input_features, output_timesteps, output_features,
        model_complexity_level='simple'):
    """Build the simple or complex ConvLSTM trajectory model."""
    _validate_model_shape(
        input_timesteps, input_features, output_timesteps, output_features
    )
    model_complexity_level = _validate_model_complexity_level(
        model_complexity_level
    )

    if model_complexity_level == 'simple':
        convolution_layers = [
            ConvLSTM2D(
                filters=64,
                kernel_size=(1, 3),
                padding='same',
                return_sequences=False,
                activation='relu',
            ),
        ]
    else:
        convolution_layers = [
            ConvLSTM2D(
                filters=64,
                kernel_size=(1, 3),
                padding='same',
                return_sequences=True,
                activation='relu',
            ),
            ConvLSTM2D(
                filters=32,
                kernel_size=(1, 3),
                padding='same',
                return_sequences=False,
                activation='relu',
            ),
        ]

    return Sequential([
        Input(shape=(input_timesteps, input_features)),
        # ConvLSTM2D expects time, height, width, and channel dimensions.
        Reshape((input_timesteps, 1, input_features, 1)),
        *convolution_layers,
        Flatten(),
        Dense(output_timesteps * output_features, activation='linear'),
        Reshape((output_timesteps, output_features)),
    ])


def _tcn_residual_block(x, filters, kernel_size, dilation_rate):
    """Apply one dilated residual TCN block."""
    residual = x

    x = Conv1D(
        filters, kernel_size, padding='causal', dilation_rate=dilation_rate
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.2)(x)

    x = Conv1D(
        filters, kernel_size, padding='causal', dilation_rate=dilation_rate
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.2)(x)

    # Match channels before adding the residual path.
    if tf.keras.backend.int_shape(residual)[-1] != filters:
        residual = Conv1D(filters, 1, padding='same')(residual)
    return Add()([residual, x])


def build_tcn_model(
        input_timesteps, input_features, output_timesteps, output_features,
        model_complexity_level='simple'):
    """Build the simple or complex TCN trajectory model."""
    _validate_model_shape(
        input_timesteps, input_features, output_timesteps, output_features
    )
    model_complexity_level = _validate_model_complexity_level(
        model_complexity_level
    )

    dilation_rates = (1, 2, 4, 8)
    if model_complexity_level == 'complex':
        dilation_rates += (16,)

    inputs = Input(shape=(input_timesteps, input_features))
    x = Conv1D(filters=64, kernel_size=1, padding='same')(inputs)
    # Increasing dilation expands the temporal receptive field.
    for dilation_rate in dilation_rates:
        x = _tcn_residual_block(x, 64, 2, dilation_rate)
    x = Flatten()(x)
    x = Dense(output_timesteps * output_features)(x)
    outputs = Reshape((output_timesteps, output_features))(x)
    return Model(inputs=inputs, outputs=outputs)


MODEL_BUILDERS = {
    'lstm': build_lstm_model,
    'gru': build_gru_model,
    'cnn': build_cnn_model,
    'convlstm': build_convlstm_model,
    'tcn': build_tcn_model,
}


get_lstm_model = build_lstm_model
get_gru_model = build_gru_model
get_cnn_model = build_cnn_model
get_convlstm_model = build_convlstm_model
get_tcn_model = build_tcn_model


def _get_model_key(model_name):
    if not isinstance(model_name, str):
        raise TypeError('model_name must be a string.')

    model_key = model_name.lower().replace('-', '').replace('_', '').strip()
    if model_key not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {AVAILABLE_MODELS}."
        )
    return model_key


def get_model_builder(model_name, model_complexity_level='simple'):
    """Return a four-argument builder for an existing training function."""
    model_key = _get_model_key(model_name)
    model_complexity_level = _validate_model_complexity_level(
        model_complexity_level
    )
    return partial(
        MODEL_BUILDERS[model_key],
        model_complexity_level=model_complexity_level,
    )


def build_trajectory_model(
        model_name, input_timesteps, input_features,
        output_timesteps, output_features, model_complexity_level='simple'):
    """Build a vessel trajectory model by name and complexity level."""
    model_key = _get_model_key(model_name)

    return MODEL_BUILDERS[model_key](
        input_timesteps,
        input_features,
        output_timesteps,
        output_features,
        model_complexity_level,
    )


class TrajectoryModelFactory:
    """Create reusable vessel trajectory prediction models."""

    @staticmethod
    def create(
            model_name, input_timesteps, input_features,
            output_timesteps, output_features,
            model_complexity_level='simple'):
        return build_trajectory_model(
            model_name,
            input_timesteps,
            input_features,
            output_timesteps,
            output_features,
            model_complexity_level,
        )


__all__ = [
    'AVAILABLE_MODELS',
    'MODEL_COMPLEXITY_LEVEL',
    'MODEL_BUILDERS',
    'TrajectoryModelFactory',
    'build_trajectory_model',
    'get_model_builder',
    'build_lstm_model',
    'build_gru_model',
    'build_cnn_model',
    'build_convlstm_model',
    'build_tcn_model',
    'get_lstm_model',
    'get_gru_model',
    'get_cnn_model',
    'get_convlstm_model',
    'get_tcn_model',
]
