import math
import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64
from scipy.stats import norm


def BSDivCallValue(sigma, S0, K, r, delta, T):
    """
    Define analytical functions based on Black Scholes formula for dividend paying stock
    """
    d1 = (math.log(S0 / K) + (r - delta + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-delta * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def EuropeanOptionPriceHurst(sigma, S0, K, r, delta, T, H=0.5, N=300, M=1000000):
    """
    Create the function to price European Call options using Monte Carlo scheme for different Hurst parameters
    """

    # pre-compute constants
    dt = T / N
    nudt = (r - delta - 0.5 * sigma ** 2) * dt
    sighdt = sigma * dt ** H

    # generate a matrix of standard normal random numbers of size MxN
    rg = Generator(PCG64())
    Z = rg.standard_normal(size=[M, N])

    # Generate stock value paths
    lnSt = np.log(S0)

    for i in range(N):
        lnSt = lnSt + nudt + sighdt * Z[:, i]
    ST = np.exp(lnSt)

    # Calculate the discounted value of the option
    disc_VT = np.maximum(0, ST - K) * np.exp(-r * T)

    return np.mean(disc_VT)


if __name__ == '__main__':
    # Set the option values
    S0 = 100.0
    K = 100.0
    r = 0.05
    delta = 0.0
    T = 1.0
    sigma = 0.2

    # Calculate the analytical BS call option price
    BS = BSDivCallValue(sigma, S0, K, r, delta, T)

    # Calculate Monte Carlo call option price using different values of N and M
    MC_H02 = EuropeanOptionPriceHurst(sigma, S0, K, r, delta, T, H=0.2, N=300, M=100000)
    MC_H05 = EuropeanOptionPriceHurst(sigma, S0, K, r, delta, T, H=0.5, N=300, M=100000)
    MC_H06 = EuropeanOptionPriceHurst(sigma, S0, K, r, delta, T, H=0.6, N=300, M=100000)
    MC_H09 = EuropeanOptionPriceHurst(sigma, S0, K, r, delta, T, H=0.9, N=300, M=100000)

    # Store results in a dataframe
    MC_results = pd.DataFrame({"Analytic Price": [BS, BS, BS, BS],
                               "N": [300, 300, 300, 300],
                               "M": [100000, 100000, 100000, 100000],
                               "H": [0.2, 0.5, 0.6, 0.9],
                               "Monte Carlo Price": [MC_H02, MC_H05, MC_H06, MC_H09]})

    print(MC_results)
