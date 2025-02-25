import numpy as np

from log_maths import _log_add
from log_maths import _log_comb

import argparse

def rdp_GM(alpha, sigma, c):
    return alpha*c*c/2/sigma/sigma

def rdp_SGM(q, alpha, sigma, c):
    current_log_sum = (alpha-1)*np.log(1-q) + np.log(alpha*q-q+1)
    for k in range(2,alpha+1):
        current_term =  _log_comb(alpha, k)+ (alpha-k)*np.log(1-q) + k*np.log(q) + \
            (k-1)*k * c * c  / 2 / sigma / sigma
        current_log_sum = _log_add(current_log_sum, current_term)
    return current_log_sum / (alpha-1)


"""
Privacy analysis for poisson subsampled Gaussian mechanism https://arxiv.org/pdf/1908.10530.pdf under unbounded-dp definition.
Conversion rule from RDP to approximate DP is from https://proceedings.neurips.cc/paper/2020/file/b53b3a3d6ab90ce0268229151c9bde11-Paper.pdf Proposition 7.
Example:
  python dpsgd_analysis.py --epochs=10 --q=0.001 --sigma_multi=4.396

q = 0.001
eps   epoch   sigma_multi  

  
"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TRAIN
    parser.add_argument("--q", help="sampling_rate per batch", type=float, default=0.1)
    parser.add_argument("--epochs", help="", type=int)
    # sigma
    parser.add_argument("--sigma_multi", help="scale of sigma_multiplier applied on gradient c", type=float, default=0.01)
    # privacy meter
    parser.add_argument("--delta", help="delta in approximate DP", type=float, default=1e-5)

    args = parser.parse_args()
    print(args)

    q = args.q
    epochs = args.epochs
    sigma = args.sigma_multi
    delta = args.delta
    rounds = int(1 / q) * epochs

    best_alpha = 0
    best_eps = np.inf
    alpha_candidates = np.array(range(2,100))
    for alpha in alpha_candidates:
        #print(alpha)
        current_eps = np.inf
        if q == 1:
            current_eps = rdp_GM(alpha, sigma, 1)
        else:
            current_eps = rdp_SGM(q,alpha, sigma, 1)
        current_eps = current_eps*rounds +\
            (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
        if current_eps < best_eps:
            best_eps = current_eps
            best_alpha = alpha

    print('Best eps: ', best_eps, ' is achieved at alpha: ', best_alpha,
        ' with delta fixed to ', delta)