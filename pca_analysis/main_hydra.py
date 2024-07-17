import argparse
import logging
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataset import get_datase
from util import gaussian_noise, prob_round_tensor, skellam_noise
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import warnings

warnings.filterwarnings("ignore")
from sklearn import preprocessing



def setup_log(name: str) -> logging.Logger:
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    filename = f"log_{name}.log"
    log_handler = logging.FileHandler(f"log/{filename}", mode="w")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(log_format)
    my_logger.addHandler(log_handler)
    return my_logger


def PCA(X, num_components):
    X_meaned = X - torch.mean(X, 0, keepdim=True)

    cov_mat = (X_meaned.t().mm(X_meaned)) / (X_meaned.shape[0] - 1)  # ??
    eigen_values, eigen_vectors = torch.linalg.eigh(cov_mat)
    sorted_index = torch.argsort(eigen_values, descending=True)
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    X_reduced = torch.matmul(eigenvector_subset.t(), X_meaned.t()).t()

    return X_reduced, eigenvector_subset


def SVD(X):
    X_meaned = X - torch.mean(X, 0, keepdim=True)

    cov_mat = (X_meaned.t().mm(X_meaned)) / (X_meaned.shape[0] - 1)  # ??
    eigen_values, eigen_vectors = torch.linalg.eigh(cov_mat)
    sorted_index = torch.argsort(eigen_values, descending=True)
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    return sorted_eigenvectors


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    log_file = "_".join(
        [
            args.setting,
            args.dataset,
            str(args.random_seed),
            str(args.skellam_mu),
            str(args.sigma),
            str(args.b),
            str(args.clipping_norm),
        ]
    )

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    logger = setup_log(log_file)

    IS_PLOT = False
    IS_ANALYSIS = True
    log_dir = "results"
    logger.info(args)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    baseline_time = time.time()

    X, y = get_datase(args.dataset, args.clipping_norm)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    y = torch.tensor(y, device=device)
    X = torch.tensor(X, device=device)
    
    logger.info("====>Load data time: %s", time.time() - baseline_time)
    baseline_time = time.time()
    file_name = "_".join(
        [
            args.setting,
            args.dataset,
            str(X.shape[0]),
            str(args.random_seed),
            str(args.skellam_mu),
            str(args.sigma),
            str(args.b),
            str(args.clipping_norm),
        ]
    )

    if os.path.exists(f"{log_dir}/{file_name}.pkl") is False:
        if args.setting == "fl":
            discretized_x = prob_round_tensor(
                X, device
            )  
            logger.info("Discretize data time: %s", time.time() - baseline_time)

            baseline_time = time.time()
            clean_x = torch.matmul(discretized_x.t(), discretized_x)
            assert clean_x.shape == (
                X.shape[1],
                X.shape[1],
            ), "multiplication error"
            tri_upper_diag_x = torch.triu(clean_x, diagonal=0)
            noise_size = tri_upper_diag_x.shape
            mu = args.skellam_mu  # for skellam value
            noise = skellam_noise(noise_size, mu, device)
            logger.info(f"mu:{mu}, skellam_mu:{args.skellam_mu}")
            tri_upper_diag_x = torch.triu(tri_upper_diag_x + noise, diagonal=0)
            noisy_cov = (
                tri_upper_diag_x
                + tri_upper_diag_x.t()  # Transposing the tensor
                - torch.diag(torch.diag(tri_upper_diag_x))  # Subtracting the diagonal
            )
            logger.info(
                "----->l2 distance between the noisy x and true x : %.5f",
                torch.norm(noisy_cov - torch.mm(X.t(), X)),
            )
            logger.info("====>Add noise time: %s", time.time() - baseline_time)
            tri_upper_diag_x.to("cpu")

        elif args.setting == "centralized":
            baseline_time = time.time()
            multiplied_x = torch.matmul(X.t(), X)
            tri_upper_diag_x = torch.triu(multiplied_x, diagonal=0)
            noise_size = tri_upper_diag_x.shape
            noise = gaussian_noise(noise_size, args.sigma, device)
            tri_upper_diag_x = torch.triu(tri_upper_diag_x + noise, diagonal=0)
            noisy_cov = (
                tri_upper_diag_x
                + tri_upper_diag_x.t()
                - torch.diag_embed(torch.diag(tri_upper_diag_x))
            )
            logger.info(
                "----->l2 distance between the noisy x and true x : %.5f",
                torch.norm(noisy_cov - torch.mm(X.t(), X)),
            )
            logger.info("====>Add noise time: %s", time.time() - baseline_time)
            tri_upper_diag_x.to("cpu")

        elif args.setting == "local":
            baseline_time = time.time()
            noise_size = X.shape
            noise = gaussian_noise(noise_size, args.sigma, device)
            noisy_x = X + noise
            noisy_cov = torch.matmul(noisy_x.t(), noisy_x)
            logger.info(
                "----->l2 distance between the noisy x and true x : %.5f",
                torch.norm(noisy_cov - torch.mm(X.t(), X)),
            )
            logger.info("====>Add noise time: %s", time.time() - baseline_time)

        baseline_time = time.time()
        logger.info(noisy_cov.shape)
        eigen_values, eigen_vectors = torch.linalg.eigh(noisy_cov)
        sorted_index = torch.argsort(eigen_values, descending=True)
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        noisy_cov.to("cpu")
        noise.to("cpu")
        sorted_eigenvalue.to("cpu")
        sorted_eigenvectors.to("cpu")

        
    else:
        if IS_ANALYSIS:
            with open(f"{log_dir}/{file_name}.pkl", "rb") as f:
                saved_results = pickle.load(f)
                sorted_eigenvalue = saved_results["eigenvalue"].to(device)
                sorted_eigenvectors = saved_results["eigenvector"].to(device)

    if IS_ANALYSIS:
        clean_sorted_eigenvectors = SVD(X)

        for num_components in [100, 200, 400, 800, 1600, 3200]:
            logger.info("====>num_components: %s", num_components)
            noisy_eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
            noisy_eigenvector_subset.to(device)
            print("noisy shape", noisy_eigenvector_subset.shape)
            logger.info("====>SVD time: %s", time.time() - baseline_time)

            baseline_time = time.time()
            clean_eigenvector_subset = clean_sorted_eigenvectors[
                :, 0:num_components
            ]  ##PCA(X, num_components=num_components)
            A = torch.matmul(X.t(), X) / X.shape[0]
            q_f = (
                torch.trace(
                    torch.matmul(
                        torch.matmul(noisy_eigenvector_subset.t(), A),
                        noisy_eigenvector_subset,
                    )
                )
                .cpu()
                .item()
            )
            q_f_clean = (
                torch.trace(
                    torch.matmul(
                        torch.matmul(clean_eigenvector_subset.t(), A),
                        clean_eigenvector_subset,
                    )
                )
                .cpu()
                .item()
            )
            logger.info("q_f:  %.8f", q_f)
            logger.info("q_clean: %.8f", q_f_clean)

            results = {
                "setting": [args.setting],  # if args.setting != "fl" else ["fl_new"],
                "dataset": [args.dataset],
                "num_samples": [X.shape[0]],
                "num_components": [num_components],
                "features": [X.shape[1]],
                "random_seed": [args.random_seed],
                "skellam_mu": [args.skellam_mu],
                "sigma": [args.sigma],
                "b": [args.b],
                "clipping_norm": [args.clipping_norm],
                "q_f": [q_f],
                "q_clean": [q_f_clean],
            }
            df = pd.DataFrame(results)
            df.to_csv("pca_results_new_alg.csv", mode="a", index=False, header=False)
            logger.info("====>Measure time: %s", time.time() - baseline_time)

        if IS_PLOT:
            X_reduced = torch.matmul(noisy_eigenvector_subset.t(), X.t()).t()
            principal_df = pd.DataFrame(X_reduced.cpu().numpy(), columns=["PC1", "PC2"])
            principal_df = pd.concat(
                [principal_df, pd.DataFrame(y)], axis=1
            )  # Assuming y is a numpy array or pandas DataFrame
            plt.figure(figsize=(6, 6))
            sns.scatterplot(
                data=principal_df,
                x="PC1",
                y="PC2",
                hue="target",
                s=60,
                palette="icefire",
            )

            # 8. also show the result wihtout noise
            clean_principal_df = pd.DataFrame(
                PCA(X, num_components)[0].cpu().numpy(), columns=["PC1", "PC2"]
            )  # Assuming PCA function is updated to use PyTorch
            clean_principal_df = pd.concat(
                [clean_principal_df, pd.DataFrame(y)], axis=1
            )
            sns.scatterplot(
                data=clean_principal_df,
                x="PC1",
                y="PC2",
                hue="target",
                s=60,
                marker="+",
            )
            plt.show()

if __name__ == "__main__":
    main()
