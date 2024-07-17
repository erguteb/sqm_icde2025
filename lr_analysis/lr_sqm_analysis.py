import numpy as np

from log_maths import _log_add
from log_maths import _log_comb

import argparse

def rdp_sk(alpha, mu_normalized, gamma,  d):
    normalized_l2_2 = 9/16 + 9*d/gamma + 36/gamma/gamma
    normalized_l1 = min(np.sqrt(d*normalized_l2_2), normalized_l2_2)
    return alpha * normalized_l2_2 / 2 / mu_normalized + min(((2*alpha-1)*normalized_l2_2+6*normalized_l1)/16/mu_normalized/mu_normalized, 3*normalized_l1/4/mu_normalized)



def rdp_sample_sk(q, alpha, mu_normalized, gamma,  d):
    current_log_sum = (alpha-1)*np.log(1-q) + np.log(alpha*q-q+1)
    for k in range(2,alpha+1):
        current_term =  _log_comb(alpha, k)+ (alpha-k)*np.log(1-q) + k*np.log(q) + \
            (k-1)*rdp_sk(k, mu_normalized, gamma, d)
        current_log_sum = _log_add(current_log_sum, current_term)
    return current_log_sum / (alpha-1)


"""
Example:
  python lr_sqm_analysis.py --epochs=30 --q=0.001 --gamma=100 --d=811 --mu_normalized=0.2639
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument("--q", help="sampling_rate per batch", type=float, default=0.1)
    parser.add_argument("--d", help="dimension of x", type=float)

    # TRAIN
    parser.add_argument("--epochs", help="", type=int)
    parser.add_argument("--gamma", help="scaling parameter", type=float, default=1)
    # NOISE
    parser.add_argument("--mu_normalized", help="normalized noise scale applied to approx grad sum", type=float, default=0.01)

    # privacy meter
    parser.add_argument("--delta", help="delta in approximate DP", type=float, default=0.00001)

    args = parser.parse_args()

    print(args)

    q = args.q
    d = args.d
    epochs = args.epochs
    gamma = args.gamma
    mu_normalized = args.mu_normalized
    mu = mu_normalized * gamma * gamma * gamma
    delta = args.delta

    rounds = int(1 / q) * epochs

    best_alpha = 0
    best_eps = np.inf
    alpha_candidates = np.array(range(2,200))
    for alpha in alpha_candidates:
        #print(alpha)
        if alpha > 2 * mu / (d+2) + 1:
            break
        current_eps = np.inf
        if q == 1:
            current_eps = rdp_sk(alpha, alpha, mu_normalized, gamma, d)
        else:
            current_eps = rdp_sample_sk(q, alpha, mu_normalized, gamma, d)
        current_eps = current_eps*rounds +\
            (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
        if current_eps < best_eps:
            best_eps = current_eps
            best_alpha = alpha

    print('Best eps: ', best_eps, ' is achieved at alpha: ', best_alpha)
    print('normalized l2_sens is:', np.sqrt(9/16 + 9*d/gamma + 36/gamma/gamma), 'overhead is:', np.sqrt(9/16 + 9*d/gamma + 36/gamma/gamma)-0.75)
    print('normalized noise scale applied to approx grad sum is', mu_normalized)
