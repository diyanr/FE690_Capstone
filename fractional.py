import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from numpy.random import Generator, PCG64
import time
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from fbm import FBM


def covariance(i, H):
    """
    Compute the covariance
    """
    if (i == 0):
        return 1
    else:
        return ((i - 1) ** (2 * H) - 2 * i ** (2 * H) + (i + 1) ** (2 * H)) / 2;


def computeE(N, H):
    """
    Compute the eigenvalues
    """
    # Create the circular matrix row
    forward = [covariance(i, H) for i in range(1, N)]
    reverse = list(reversed(forward))
    row = [covariance(0, H)] + forward + [0] + reverse

    # return the eigenvalues
    return fft(row).real


def computeW(eigenvals, N, H):
    """
    Compute W=Q*V
    """
    # generate two standard normal random numbers of size N
    rg = Generator(PCG64())
    V1 = rg.standard_normal(size=N)
    V2 = rg.standard_normal(size=N)

    # create a vector of complex weight W
    W = np.zeros(2 * N, dtype=complex)
    for i in range(2 * N):
        if i == 0:
            W[i] = np.sqrt(eigenvals[i] / (2 * N)) * V1[i]
        elif i < N:
            W[i] = np.sqrt(eigenvals[i] / (4 * N)) * (V1[i] + 1j * V2[i])
        elif i == N:
            W[i] = np.sqrt(eigenvals[i] / (2 * N)) * V2[0]
        else:
            W[i] = np.sqrt(eigenvals[i] / (4 * N)) * (V1[2 * N - i] - 1j * V2[2 * N - i])

    return W


def fbmDH(n, H, T):
    """
    Calculate a sample path using the Davis Harte method
    for fractional Brownian motion with Hurst parameter H
    """

    # Calculate N as a power of 2 greater than n to make fft process faster
    N = int(math.pow(2, math.ceil(math.log2(n))))
    scale = (T / N) ** H

    # STEP 1: Compute the eigenvalues E of the circulant matrix
    E = computeE(N, H)

    # STEP 2: Compute  W = Q*V
    W = computeW(E, N, H)

    # STEP 3: Compute Z = QE^(1/2)W
    Z = fft(W)

    # take first n samples as scaled fractional Gaussian Noise
    fgn = scale * Z[: n].real

    # return the fractional Brownian motion
    return np.insert(fgn.cumsum(), [0], 0)


def hurst_f(input_ts, lags_to_test=100):
    """
    interpretation of return value
    hurst < 0.5 - input_ts is mean reverting
    hurst = 0.5 - input_ts is effectively random/geometric brownian motion
    hurst > 0.5 - input_ts is trending
    """
    tau = []
    lagvec = []
    #  Step through the different lags
    for lag in range(2, lags_to_test):
        #  produce price difference with lag
        pp = np.subtract(input_ts[lag:], input_ts[:-lag])
        #  Write the different lags into a vector
        lagvec.append(lag)
        #  Calculate the variance of the difference vector
        tau.append(np.sqrt(np.std(pp)))
        #  linear fit to double-log graph (gives power)
    m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
    # calculate hurst
    hurst_ind = m[0] * 2
    return hurst_ind


if __name__ == '__main__':
    fig = plt.figure(figsize=(20, 40))
    fig.suptitle('Fractional Brownian Motion (fBM)', fontsize=30)

    ax1 = plt.subplot2grid((4, 1), (0, 0))
    ax1.plot(fbmDH(n=2000, H=0.2, T=1), color='#1f77b4')
    ax1.set_title('fBM using code for H = 0.2', fontsize=20)

    ax2 = plt.subplot2grid((4, 1), (1, 0))
    ax2.plot(fbmDH(n=2000, H=0.5, T=1), color='#1f77b4')
    ax2.set_title('fBM using code for H = 0.5', fontsize=20)

    ax3 = plt.subplot2grid((4, 1), (2, 0))
    ax3.plot(fbmDH(n=2000, H=0.6, T=1), color='#1f77b4')
    ax3.set_title('fBM using code for H = 0.6', fontsize=20)

    ax4 = plt.subplot2grid((4, 1), (3, 0))
    ax4.plot(fbmDH(n=2000, H=0.9, T=1), color='#1f77b4')
    ax4.set_title('fBM using code for H = 0.9', fontsize=20)

    plt.savefig(r"\Users\diyan\Documents\WQU\Capstone\Develop\fbm.png")
