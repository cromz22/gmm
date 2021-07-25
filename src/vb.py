import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import digamma, logsumexp
import numpy as np
from numpy.linalg import det, inv
import pickle
import csv
import sys


def vb_e_step(x, alpha, nu, W, beta):
    """
    Calculate expectations and then responsibilities
    """
    E_ln_pi_ks = digamma(alpha) - digamma(sum(alpha))

    E_ln_Lambda_ks = []
    for k in range(K):
        digamma_ary = []
        for i in range(D):
            digamma_ary.append(
                digamma((nu[k] + 1 - i) / 2)
            )
        E_ln_Lambda_k = sum(digamma_ary) + D * np.log(2) + np.log(det(W[k]))
        E_ln_Lambda_ks.append(E_ln_Lambda_k)

    ln_rho = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            vec = (x[n] - m[k]).reshape(D, 1)
            E_muk_Lambdak = D / beta[k] + nu[k] * vec.T @ W[k] @ vec

            ln_rho[n][k] = E_ln_pi_ks[k] + 0.5 * E_ln_Lambda_ks[k] - \
                D * 0.5 * np.log(2*np.pi) - 0.5 * E_muk_Lambdak

    ln_r = ln_rho - logsumexp(ln_rho, axis=1).reshape(N, 1)
    # logsumexp(a) = log(sum(exp(a)))

    r = np.exp(ln_r)  # (N, K)
    return r


def vb_m_step(x, r, alpha, beta, m, W, nu, pi):
    """
    Update parameters
    """
    N_k = np.sum(r, axis=0).reshape(K, 1)  # (K, 1)

    x_k_bar = r.T @ x / N_k
    # (K, N) x (N, D) -> (K, D); (K, D) / (K, 1) -> (K, D)

    S_k_ary = np.zeros((N, K, D, D))
    for n in range(N):
        for k in range(K):
            vec = (x[n] - x_k_bar[k]).reshape(D, 1)
            S_k_ary[n][k] = r[n][k] * vec @ vec.T
    S_k = np.sum(S_k_ary, axis=0) / N_k.reshape(K, 1, 1)  # (K, D, D)

    alpha = alpha0 + N_k  # (K, 1)

    beta = beta0 + N_k  # (K, 1)

    # m: (K, D)
    for k in range(K):
        m[k] = (beta0 * m0 + N_k[k] * x_k_bar[k]) / beta[k]

    # W: (K, D, D)
    for k in range(K):
        term1 = inv(W0)
        term2 = N_k[k] * S_k[k]
        vec = (x_k_bar[k] - m0).reshape(D, 1)
        term3 = (beta0 * N_k[k] / (beta0 + N_k[k])) * vec @ vec.T
        W[k] = inv(term1 + term2 + term3)

    nu = nu0 + N_k  # (K, 1)

    pi = alpha / np.sum(alpha)

    return alpha, beta, m, W, nu, pi


def evaluate_log_likelihood(x, mu, Sigma, pi, nu):
    """
        evaluate the log likelihood of getting x given parameters
    input:
        x: (N, D)
        mu: (K, D) From the parameter of Gaussian-Wishart Distribution
        Sigma: (K, D, D) From the parameter of Gaussian-Wishart Distribution
        pi: (K, )
    output: log likelihood (scalar)
    """
    for k in range(K):
        Sigma[k] = inv(nu[k] * W[k])

    gaussians = np.array([
        pi[k] * multivariate_normal.pdf(x, mean=mu[k], cov=Sigma[k])
        for k in range(K)
    ])  # (K, N)

    gm = sum(gaussians)

    return sum(np.log(gm))


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 4:
        print("Usage: python vb.py x.csv z.csv params.dat")
        exit(1)

    # load data
    df = pd.read_csv("x.csv", header=None)
    x = df.to_numpy()
    N, D = x.shape

    K = 4

    alpha0 = 0.1
    beta0 = 1.0

    m0 = np.random.randn(D)
    W0 = np.eye(D)
    nu0 = D

    alpha = np.full(K, alpha0)
    beta = np.full(K, beta0)
    m = np.random.randn(K, D)
    W = np.tile(W0, (K, 1, 1))  # [[W0], [W0], [W0]]
    nu = np.full(K, nu0)
    Sigma = np.zeros((K, D, D))
    for k in range(K):
        Sigma[k] = inv(nu[k] * W[k])
    mu = m
    pi = np.array([1 / K for _k in range(K)])

    threshold = 0.01
    max_iter = 100

    log_likelihood_record = []
    log_likelihood_record.append(evaluate_log_likelihood(x, mu, Sigma, pi, nu))
    for i in range(max_iter):
        r = vb_e_step(x, alpha, nu, W, beta)
        alpha, beta, m, W, nu, pi = vb_m_step(x, r, alpha, beta, m, W, nu, pi)
        ll = evaluate_log_likelihood(x, mu, Sigma, pi, nu)
        log_likelihood_record.append(ll)
        print(f"log likelihood: {ll:.3f}")
        diff = ll - log_likelihood_record[i]
        if diff < threshold:
            print(f"converged with {i} iterations")
            break

    with open("z.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(r)

    with open("params.dat", "wb") as f:
        pickle.dump(alpha, f)
        pickle.dump(beta, f)
        pickle.dump(m, f)
        pickle.dump(W, f)
        pickle.dump(nu, f)
        pickle.dump(pi, f)
