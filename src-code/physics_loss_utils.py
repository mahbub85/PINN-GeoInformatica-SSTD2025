"""
Author: Md Mahbub Alam
File: Physics-informed loss functions for vessel trajectory prediction

Loss functions:
    1. First-Order Euler - Small-Angle Approximation
    2. First-Order Euler - Great-Circle Approximation
    3. Second-Order Heun - Small-Angle Approximation
    4. Second-Order Heun - Great-Circle Approximation
"""

import math

import numpy as np
import tensorflow as tf


__all__ = [
    'physics_loss_euler_small_angle',
    'physics_loss_euler_great_circle',
    'physics_loss_heun_small_angle',
    'physics_loss_heun_great_circle',
]


def physics_loss_euler_small_angle(
        y_true, y_pred, sog, cog, min_max, R=6371000.0, dt=120.0):
    """First-order Euler physics loss using a small-angle approximation."""
    lat_min, lat_max = min_max['lat']
    lon_min, lon_max = min_max['lon']
    sog_min, sog_max = min_max['sog']
    cog_min, cog_max = min_max['cog']

    pred_lat = y_pred[..., 0] * (lat_max - lat_min) + lat_min
    pred_lon = y_pred[..., 1] * (lon_max - lon_min) + lon_min

    sog_denorm = tf.squeeze(
        sog * (sog_max - sog_min) + sog_min, axis=-1
    )
    cog_denorm = tf.squeeze(
        cog * (cog_max - cog_min) + cog_min, axis=-1
    )

    delta_lat_pred = pred_lat[:, 1:] - pred_lat[:, :-1]
    delta_lon_pred = pred_lon[:, 1:] - pred_lon[:, :-1]

    sog_t = sog_denorm[:, :-1]
    cog_t = cog_denorm[:, :-1]
    lat_t = pred_lat[:, :-1]

    cog_t_rad = cog_t * tf.constant(
        math.pi / 180.0, dtype=cog_t.dtype
    )
    lat_t_rad = lat_t * tf.constant(
        math.pi / 180.0, dtype=lat_t.dtype
    )

    deg_per_radian = 180.0 / np.pi
    factor = (dt / R) * deg_per_radian

    expected_delta_lat = sog_t * tf.cos(cog_t_rad) * factor
    expected_delta_lon = (
        sog_t * tf.sin(cog_t_rad) * factor / tf.cos(lat_t_rad)
    )

    residual_lat = delta_lat_pred - expected_delta_lat
    residual_lon = delta_lon_pred - expected_delta_lon

    residual_lat_norm = residual_lat / (lat_max - lat_min)
    residual_lon_norm = residual_lon / (lon_max - lon_min)

    return tf.reduce_mean(
        tf.square(residual_lat_norm) + tf.square(residual_lon_norm)
    )


def physics_loss_euler_great_circle(
        y_true, y_pred, sog, cog, acceleration, cograte, min_max,
        R=6371000.0, dt=120.0):
    """First-order Euler physics loss using a great-circle approximation."""
    lat_min, lat_max = min_max['lat']
    lon_min, lon_max = min_max['lon']
    sog_min, sog_max = min_max['sog']
    cog_min, cog_max = min_max['cog']
    acc_min, acc_max = min_max['acceleration']
    cgr_min, cgr_max = min_max['cograte']

    pred_lat = y_pred[..., 0] * (lat_max - lat_min) + lat_min
    pred_lon = y_pred[..., 1] * (lon_max - lon_min) + lon_min

    sog_denorm = tf.squeeze(
        sog * (sog_max - sog_min) + sog_min, axis=-1
    )
    cog_denorm = tf.squeeze(
        cog * (cog_max - cog_min) + cog_min, axis=-1
    )
    acceleration_denorm = tf.squeeze(
        acceleration * (acc_max - acc_min) + acc_min, axis=-1
    )
    cograte_denorm = tf.squeeze(
        cograte * (cgr_max - cgr_min) + cgr_min, axis=-1
    )

    delta_lat_pred = pred_lat[:, 1:] - pred_lat[:, :-1]
    delta_lon_pred = pred_lon[:, 1:] - pred_lon[:, :-1]

    sog_t = sog_denorm[:, :-1]
    cog_t = cog_denorm[:, :-1]
    acceleration_t = acceleration_denorm[:, :-1]
    cograte_t = cograte_denorm[:, :-1]
    lat_t = pred_lat[:, :-1]

    cog_t_rad = cog_t * tf.constant(
        math.pi / 180.0, dtype=cog_t.dtype
    )
    lat_t_rad = lat_t * tf.constant(
        math.pi / 180.0, dtype=lat_t.dtype
    )
    cograte_t_rad = cograte_t * tf.constant(
        math.pi / 180.0, dtype=cograte_t.dtype
    )

    dt_seconds = tf.constant(dt, dtype=lat_t.dtype)
    R_meters = tf.constant(R, dtype=lat_t.dtype)

    # Midpoint COG and SOG
    cog_mid_rad = cog_t_rad + 0.5 * cograte_t_rad * dt_seconds
    angular_distance = (
        (sog_t + 0.5 * acceleration_t * dt_seconds)
        * dt_seconds / R_meters
    )

    sin_lat_cos_d = tf.sin(lat_t_rad) * tf.cos(angular_distance)
    cos_lat_sin_d_cos_cog = (
        tf.cos(lat_t_rad)
        * tf.sin(angular_distance)
        * tf.cos(cog_mid_rad)
    )
    next_lat_arg = tf.clip_by_value(
        sin_lat_cos_d + cos_lat_sin_d_cos_cog, -1.0, 1.0
    )
    next_lat_rad = tf.asin(next_lat_arg)

    lon_numerator = (
        tf.sin(cog_mid_rad)
        * tf.sin(angular_distance)
        * tf.cos(lat_t_rad)
    )
    lon_denominator = (
        tf.cos(angular_distance)
        - tf.sin(lat_t_rad) * tf.sin(next_lat_rad)
    )

    deg_per_radian = tf.constant(180.0 / np.pi, dtype=lat_t.dtype)
    expected_delta_lat = (
        next_lat_rad - lat_t_rad
    ) * deg_per_radian
    expected_delta_lon = tf.atan2(
        lon_numerator, lon_denominator
    ) * deg_per_radian

    residual_lat = delta_lat_pred - expected_delta_lat
    residual_lon = delta_lon_pred - expected_delta_lon

    residual_lat_norm = residual_lat / (lat_max - lat_min)
    residual_lon_norm = residual_lon / (lon_max - lon_min)

    return tf.reduce_mean(
        tf.square(residual_lat_norm) + tf.square(residual_lon_norm)
    )


def _small_angle_derivatives(
        lat, lon, sog, cog, acceleration, cograte, dt, R):
    """Calculate small-angle lat/lon rates."""
    dtype = lat.dtype
    dt = tf.cast(dt, dtype)
    R = tf.cast(R, dtype)
    deg_per_radian = tf.cast(180.0 / np.pi, dtype)

    cog_rad = cog * tf.cast(np.pi / 180.0, dtype)
    lat_rad = lat * tf.cast(np.pi / 180.0, dtype)
    cograte_rad = cograte * tf.cast(np.pi / 180.0, dtype)

    # Midpoint COG and SOG
    cog_mid_rad = cog_rad + 0.5 * cograte_rad * dt
    expected_sog = sog + 0.5 * acceleration * dt

    dlat_dt = expected_sog * tf.cos(cog_mid_rad) * deg_per_radian / R
    dlon_dt = (
        expected_sog * tf.sin(cog_mid_rad)
        * deg_per_radian / (R * tf.cos(lat_rad))
    )
    return dlat_dt, dlon_dt


def _great_circle_derivatives(
        lat, lon, sog, cog, acceleration, cograte, dt, R):
    """Calculate great-circle lat/lon rates."""
    dtype = lat.dtype
    dt = tf.cast(dt, dtype)
    R = tf.cast(R, dtype)
    deg_per_radian = tf.cast(180.0 / np.pi, dtype)

    lat_rad = lat * tf.cast(np.pi / 180.0, dtype)
    cog_rad = cog * tf.cast(np.pi / 180.0, dtype)
    cograte_rad = cograte * tf.cast(np.pi / 180.0, dtype)

    # Midpoint COG and SOG
    cog_mid_rad = cog_rad + 0.5 * cograte_rad * dt
    expected_sog = sog + 0.5 * acceleration * dt
    angular_distance = expected_sog * dt / R

    next_lat_arg = (
        tf.sin(lat_rad) * tf.cos(angular_distance)
        + tf.cos(lat_rad) * tf.sin(angular_distance) * tf.cos(cog_mid_rad)
    )
    next_lat_rad = tf.asin(
        tf.clip_by_value(next_lat_arg, -1.0, 1.0)
    )

    lon_numerator = (
        tf.sin(cog_mid_rad)
        * tf.sin(angular_distance)
        * tf.cos(lat_rad)
    )
    lon_denominator = (
        tf.cos(angular_distance)
        - tf.sin(lat_rad) * tf.sin(next_lat_rad)
    )

    delta_lat = (next_lat_rad - lat_rad) * deg_per_radian
    delta_lon = tf.atan2(
        lon_numerator, lon_denominator
    ) * deg_per_radian
    return delta_lat / dt, delta_lon / dt


def _heun_step(
        lat_t, lon_t, sog_t, cog_t, acceleration_t, cograte_t,
        dt, R, derivatives):
    """Calculate one Heun predictor-corrector step."""
    dtype = lat_t.dtype
    dt = tf.cast(dt, dtype)

    # Stage 1
    dlat_dt_t, dlon_dt_t = derivatives(
        lat_t, lon_t, sog_t, cog_t,
        acceleration_t, cograte_t, dt, R
    )

    # Euler predictor
    lat_pred_end = lat_t + dlat_dt_t * dt
    lon_pred_end = lon_t + dlon_dt_t * dt
    sog_pred_end = sog_t + acceleration_t * dt
    cog_pred_end = cog_t + cograte_t * dt

    # Stage 2
    dlat_dt_end, dlon_dt_end = derivatives(
        lat_pred_end, lon_pred_end, sog_pred_end, cog_pred_end,
        acceleration_t, cograte_t, dt, R
    )

    delta_lat = 0.5 * (dlat_dt_t + dlat_dt_end) * dt
    delta_lon = 0.5 * (dlon_dt_t + dlon_dt_end) * dt
    return delta_lat, delta_lon


def _physics_loss_heun(
        y_pred, sog, cog, acceleration, cograte, min_max,
        derivatives, R, dt):
    """Calculate Heun residual loss for the selected geometry."""
    lat_min, lat_max = min_max['lat']
    lon_min, lon_max = min_max['lon']
    sog_min, sog_max = min_max['sog']
    cog_min, cog_max = min_max['cog']
    acc_min, acc_max = min_max['acceleration']
    cgr_min, cgr_max = min_max['cograte']

    pred_lat = y_pred[..., 0] * (lat_max - lat_min) + lat_min
    pred_lon = y_pred[..., 1] * (lon_max - lon_min) + lon_min

    sog_denorm = tf.squeeze(
        sog * (sog_max - sog_min) + sog_min, axis=-1
    )
    cog_denorm = tf.squeeze(
        cog * (cog_max - cog_min) + cog_min, axis=-1
    )
    acceleration_denorm = tf.squeeze(
        acceleration * (acc_max - acc_min) + acc_min, axis=-1
    )
    cograte_denorm = tf.squeeze(
        cograte * (cgr_max - cgr_min) + cgr_min, axis=-1
    )

    delta_lat_pred = pred_lat[:, 1:] - pred_lat[:, :-1]
    delta_lon_pred = pred_lon[:, 1:] - pred_lon[:, :-1]

    expected_delta_lat, expected_delta_lon = _heun_step(
        pred_lat[:, :-1],
        pred_lon[:, :-1],
        sog_denorm[:, :-1],
        cog_denorm[:, :-1],
        acceleration_denorm[:, :-1],
        cograte_denorm[:, :-1],
        dt, R, derivatives
    )

    residual_lat = delta_lat_pred - expected_delta_lat
    residual_lon = delta_lon_pred - expected_delta_lon

    residual_lat_norm = residual_lat / (lat_max - lat_min)
    residual_lon_norm = residual_lon / (lon_max - lon_min)

    return tf.reduce_mean(
        tf.square(residual_lat_norm) + tf.square(residual_lon_norm)
    )


def physics_loss_heun_small_angle(
        y_true, y_pred, sog, cog, acceleration, cograte, min_max,
        R=6371000.0, dt=120.0):
    """Second-order Heun physics loss using a small-angle approximation."""
    return _physics_loss_heun(
        y_pred, sog, cog, acceleration, cograte, min_max,
        _small_angle_derivatives, R, dt
    )


def physics_loss_heun_great_circle(
        y_true, y_pred, sog, cog, acceleration, cograte, min_max,
        R=6371000.0, dt=120.0):
    """Second-order Heun physics loss using a great-circle approximation."""
    return _physics_loss_heun(
        y_pred, sog, cog, acceleration, cograte, min_max,
        _great_circle_derivatives, R, dt
    )
