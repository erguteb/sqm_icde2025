import logging
import os
import pickle
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
from folktables import ACSDataSource, ACSIncome, generate_categories, ACSEmployment, ACSPublicCoverage

def clipping_array(data: np.ndarray, clipping_norm: float = 1):
    """
    Clips the data in the array
    """
    t = tf.constant(data, dtype=tf.float32)
    clipped_data = tf.clip_by_norm(t, clipping_norm, axes=1)
    return clipped_data.numpy()



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


def load_data(split_size=0.1, row_clip=1, gaussian_sigma=0, state='CA', flip_y=0):
    baseline_time = time.time()
    if gaussian_sigma==0 and flip_y==0:
        # this is the same as before where we do not have any guassian noise added to the features.
        file_name = f"processed_dataset/{row_clip}_{split_size}_{state}.pkl"
    file_name = f"processed_dataset/{row_clip}_{split_size}_{gaussian_sigma}_{state}_flip_{flip_y}.pkl"
    if os.path.exists(file_name) is False:
        # TODO: load different datasets here   
        """
        this code does not convert the categorical data to one hot but just factorize it 
        size: (195665,10)
        """
        data_source = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person"
        )
        ca_data = data_source.get_data(states=[state], download=True)
        definition_df = data_source.get_definitions(download=True)
        datafun = ACSIncome
        categories = generate_categories(
        features=datafun.features, definition_df=definition_df
        )
        X, y, _ = datafun.df_to_pandas(
            ca_data, categories=categories
        )
        X["label"] = y
        X = X.dropna()
        assert not X.isnull().values.any(), "NAN in the dataset"
        Y = X.iloc[:, -1].to_numpy(dtype=float)
        X = X.iloc[:, 0:-1].to_numpy(dtype=float)
        
        # convert the pandas to array 
        normalized_X = normalize_array(X, 'mean')
        logging.info(f"Normalizing the data costs: %.5f", time.time()-baseline_time)
        baseline_time = time.time()
        clipped_X = clipping_array(normalized_X, row_clip)
        logging.info(f"Clipping the data costs: %.5f", time.time()-baseline_time)
        baseline_time = time.time()
        # split to train and test
        X_train, X_test, Y_train, Y_test = train_test_split(clipped_X, Y, test_size=split_size,shuffle=True)
        print('shape of X_train', X_train.shape)
        print('shape of X_test', X_test.shape)
        
        # convert to tensor
        X_train_tensor = torch.tensor(X_train).float()
        Y_train_tensor = torch.tensor(Y_train).float()
        X_test_tensor = torch.tensor(X_test).float()
        Y_test_tensor = torch.tensor(Y_test).float()

        if gaussian_sigma > 0:
            print('perturb data')
            X_train_tensor = X_train_tensor + torch.normal(mean=0, std=gaussian_sigma, size=X_train_tensor.size())
            X_train_tensor = normalize_array(X_train_tensor, 'mean')
           

            flip_mask = torch.rand(Y_train_tensor.size()) < flip_y
            print(flip_mask)
            print(Y_train_tensor[flip_mask])
            Y_train_tensor[flip_mask] = 1 - Y_train_tensor[flip_mask]
            Y_train_tensor = Y_train_tensor.float()
            
            print(Y_train_tensor[flip_mask])
            
        train_tensor = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
        test_tensor = data_utils.TensorDataset(X_test_tensor, Y_test_tensor)
        logging.info(f"Covert to tensor costs: %.5f", time.time()-baseline_time)

        
        with open(file_name,"wb") as f:
            pickle.dump((train_tensor, test_tensor, X_train.shape[0], X_train.shape[1], 1),f)    
        return train_tensor, test_tensor, X_train.shape[0], X_train.shape[1], 1
    else:
        with open(file_name, "rb") as f:
            train_tensor, test_tensor, n, d_in, d_out = pickle.load(f)
        
        print('shape of X_train', train_tensor.tensors[0].shape, n, d_in, d_out)
        return train_tensor, test_tensor, n, d_in, d_out
    
