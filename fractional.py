import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator, PCG64
from scipy.fftpack import fft


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


def computeW(eigenvals, N):
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
    W = computeW(E, N)

    # STEP 3: Compute Z = QE^(1/2)W
    Z = fft(W)

    # take first n samples as scaled fractional Gaussian Noise
    fgn = scale * Z[: n].real

    # return the fractional Brownian motion
    return np.insert(fgn.cumsum(), [0], 0)


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
