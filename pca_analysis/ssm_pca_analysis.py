import numpy as np

import argparse

def rdp_sk(alpha, l2, l1, mu):
    sum = alpha/4/mu * l2*l2 
    sum = sum + min(((2*alpha)*l2*l2+6*l1)/16/mu/mu, 3*l1/4/mu)
    return sum

"""
Example:
  python ssm_pca_analysis_gamma.py --mu_left=0 --mu_right=1000 --prec=0.00001 --d=816 --gamma=4

Summarization of datasets:
KDDCUP     d=117, n=95666
ACSIncome  d=816, n=145585
gen        d=20531, n=801
citeseer   d=3703, n=2110
"""

def compute_best_eps(l2, l1, mu, delta):
    best_eps = np.inf
    alpha_candidates = np.array(range(2,200))
    for alpha in alpha_candidates:
        current_eps = rdp_sk(alpha, l2, l1, mu) 
        current_eps = current_eps + (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
        if current_eps < best_eps:
            best_eps = current_eps
    return best_eps

def compute_best_eps_client(l2, l1, mu, delta, N):
    best_eps = np.inf
    alpha_candidates = np.array(range(2,200))
    for alpha in alpha_candidates:
        current_eps = rdp_sk(alpha, 2*l2, 2*l1, mu*(N-1)/N) 
        current_eps = current_eps + (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
        if current_eps < best_eps:
            best_eps = current_eps
    return best_eps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # DATA
    parser.add_argument("--d", help="dimension of x", type=float)

    # Privacy
    parser.add_argument("--delta", help="target delta", type=float, default=0.00001)
    parser.add_argument("--mu_right", type=float, default=1000)
    parser.add_argument("--mu_left", type=float, default=0.0001)
    parser.add_argument("--prec", help="relative prec", type=float, default=0.0000001)
    parser.add_argument("--b", help="log of the scale parameter gamma", type=float, default=1)

    args = parser.parse_args()

    print(args)

    d = args.d
    gamma = 2 ** args.b
    print(f'b is {args.b}, gamma is:{gamma}')
    delta = args.delta
    # eps_array = np.array([0.25,1,2,4,8])
    eps_array = np.array([4, 8, 16, 32, 64])

    l2 = gamma**2 + d 
    l1 = min(np.sqrt(d)*l2, l2*l2)
    l2 = l2 / (gamma**2)
    l1 = l1 / (gamma**2)
    print(f'l2 is: {l2}, l1 is: {l1}')
        
    mu_array = []
    
    for eps in eps_array:
        mu_right = args.mu_right * l2 * l2
        mu_left = args.mu_left * l2 * l2
        left_eps = compute_best_eps(l2, l1, mu_left, delta)
        right_eps = compute_best_eps(l2, l1, mu_right, delta)
        if right_eps > eps:
            print('mu_right is too small for eps =', eps)
        elif left_eps < eps:
            print('mu_left is too large for eps =', eps)
        else:
            while True:
                current_mu = (mu_left + mu_right) / 2
                cur_eps = compute_best_eps(l2, l1, current_mu, delta)
                rel_err = np.abs(cur_eps-eps) / eps
                if np.abs(rel_err) <= args.prec:
                    print('best mu found for eps =', eps, ' rel err is ', rel_err)
                    print('N  client level DP eps\n=============')
                    for N in np.array([10,20,40,80,160,320,640]):
                        client_eps = compute_best_eps_client(l2, l1, current_mu, delta, N)
                        print(f'{N}, {client_eps}')
                    mu_array.append(current_mu)
                    break
                elif cur_eps > eps:
                    mu_left = current_mu
                else:
                    mu_right = current_mu
    
    mu_array = np.array(mu_array)
    np.set_printoptions(precision=3)
    print(mu_array)
