import math
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import torch


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df


def normalized_df(data: pd.DataFrame, features: list = None, type: str = "mean"):
    """
    Normalizes the data in the dataframe
    """
    if features is None:
        features = data.columns
    if type == "minmax":
        for feature in features:
            min_value = data[feature].min()
            max_value = data[feature].max()
            data[feature] = (data[feature] - min_value) / (max_value - min_value + 1e-8)
    elif type == "mean":
        for feature in features:
            mean = data[feature].mean()
            std = data[feature].std()
            data[feature] = (data[feature] - mean) / (std + 1e-8)
    return data


def normalize_array(data: np.ndarray, type: str):
    if type == "minmax":
        min_value = data.min(axis=0)
        max_value = data.max(axis=0)
        data = (data - min_value) / (max_value - min_value + 1e-8)
    elif type == "mean":
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        data = (data - mean) / (std + 1e-8)

    return data


def clipping_array(data: np.ndarray, clipping_norm: float = 1):
    """
    Clips the data in the array
    """
    t = tf.constant(data, dtype=tf.float32)
    clipped_data = tf.clip_by_norm(t, clipping_norm, axes=1)
    return clipped_data.numpy()


def clipping_features(x: pd.DataFrame, clipping_norm: float = 1):
    """
    Clips the data in the dataframe
    """
    clipped_x = x.apply(
        lambda row: pd.Series([i / np.linalg.norm(row) for i in row])
        if np.linalg.norm(row) > clipping_norm
        else pd.Series([i for i in row]),
        axis=1,
    )
    clipped_x = clipped_x.iloc[:, 0 : len(x.columns)]
    return clipped_x


def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = np.random.random(x.shape) < x - x.astype(int)
    round_func = math.ceil if is_up else math.floor
    return sign * round_func(x)


def prob_round_arry(x):
    floor = np.floor(x)
    p_ceil = x - floor
    while True:
        random_nums = np.random.random(size=x.shape)
        choice_floor = (random_nums > p_ceil).astype(np.float32)
        discrete_tensor = choice_floor * floor + (1.0 - choice_floor) * (floor + 1.0)
        return discrete_tensor.astype(np.int64)


def prob_round_tensor(x, device):
    floor = torch.floor(x)
    p_ceil = x - floor
    random_nums = torch.rand_like(
        x, device=device
    )  # Generates random numbers similar to the shape of x on GPU
    choice_floor = (
        random_nums > p_ceil
    ).float()  # Convert boolean tensor to float tensor
    discrete_tensor = choice_floor * floor + (1.0 - choice_floor) * (floor + 1.0)
    return discrete_tensor#.int()  # Convert to int64 tensor


def prob_round_with_prec(x, prec=0):
    fixup = np.sign(x) * 10**prec
    x *= fixup
    is_up = random.random() < x - int(x)
    round_func = math.ceil if is_up else math.floor
    return round_func(x) / fixup



def gaussian_noise(size, sigma, device):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise = torch.normal(mean=0., std=sigma, size=size, device=device)
    return noise

def skellam_noise(size, poisson_mu, device):
    noise = torch.poisson(torch.ones(size, device=device) * poisson_mu) - torch.poisson(
        torch.ones(size, device=device) * poisson_mu
    )
    return noise
