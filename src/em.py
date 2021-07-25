import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import pickle
import csv
import sys


def gaussian_mixture(x, mu_ks, sigma_ks, pi_ks):
    """
    input:
        x: (N, D)
        mu_ks: (K, D)
        sigma_ks: (K, D, D)
        pi_ks: (K, )
    output: gaussian mixture (N, )
    """
    gaussians = np.array([
        pi_ks[k] * multivariate_normal.pdf(x, mean=mu_ks[k], cov=sigma_ks[k])
        for k in range(K)
    ])  # (K, N)
    return sum(gaussians)  # (N, )


def evaluate_log_likelihood(x, mu_ks, sigma_ks, pi_ks):
    gm = gaussian_mixture(x, mu_ks, sigma_ks, pi_ks)
    return sum(np.log(gm))


def e_step(x, mu_ks, sigma_ks, pi_ks):
    """
    evaluate the responsibilities
    """
    gaussians = np.array([pi_ks[k] * multivariate_normal.pdf(x,
                                                             mean=mu_ks[k], cov=sigma_ks[k]) for k in range(K)])
    gm = gaussian_mixture(x, mu_ks, sigma_ks, pi_ks)
    gammas = gaussians / gm  # (K, N)
    return gammas


def m_step(x, gammas):
    """
    re-estimate the parameters
    """
    mu_ks = []
    sigma_ks = []
    pi_ks = []

    for k in range(K):

        N_k = sum(gammas[k])

        # mu
        ary_mu = []
        for n in range(N):
            ary_mu.append(gammas[k][n] * x[n])
        mu_k = sum(ary_mu) / N_k  # (D, )
        mu_ks.append(mu_k)

        # sigma
        ary_sigma = []
        for n in range(N):
            vec_sigma = x[n] - mu_k
            ary_sigma.append(
                gammas[k][n] * np.dot(vec_sigma.reshape(D, 1),
                                      vec_sigma.reshape(1, D))
            )
        sigma_k = sum(ary_sigma) / N_k  # (D, D)
        sigma_ks.append(sigma_k)

        # pi
        pi_k = N_k / N  # scalar
        pi_ks.append(pi_k)

    mu_ks = np.array(mu_ks)
    sigma_ks = np.array(sigma_ks)
    pi_ks = np.array(pi_ks)

    return mu_ks, sigma_ks, pi_ks


def init_params_rand():
    mu_ks = np.random.randn(K, D)
    s0 = np.eye(D)
    sigma_ks = np.tile(s0, (K, 1, 1))
    pi_ks = np.array([1 / K for k in range(K)])
    return mu_ks, sigma_ks, pi_ks


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 4:
        print("Usage: python em.py x.csv z.csv params.dat")
        exit(1)

    # load data
    df = pd.read_csv(args[1], header=None)
    x = df.to_numpy()
    N, D = x.shape

    K = 4

    # initialize parameters
    mu_ks, sigma_ks, pi_ks = init_params_rand()

    threshold = 0.01
    max_iter = 100

    log_likelihood_record = []
    log_likelihood_record.append(
        evaluate_log_likelihood(x, mu_ks, sigma_ks, pi_ks))

    for i in range(max_iter):
        gammas = e_step(x, mu_ks, sigma_ks, pi_ks)
        mu_ks, sigma_ks, pi_ks = m_step(x, gammas)
        ll = evaluate_log_likelihood(x, mu_ks, sigma_ks, pi_ks)
        log_likelihood_record.append(ll)
        print(f"log likelihood: {ll:.3f}")
        diff = ll - log_likelihood_record[i]
        if diff < threshold:
            print(f"converged with {i} iterations")
            break

    with open(args[2], "w") as f:
        writer = csv.writer(f)
        writer.writerows(gammas.T)

    with open(args[3], "wb") as f:
        pickle.dump(mu_ks, f)
        pickle.dump(sigma_ks, f)
        pickle.dump(pi_ks, f)
