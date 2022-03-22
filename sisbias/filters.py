"""Filters for cleaning experimental data."""

import numpy as np


def gauss_conv(x, sigma=10, ext_x=3):
    """Smooth data using a Gaussian convolution filter.

    Args:
        x (ndarray): noisy data to filter
        sigma (float): std. dev. of Gaussian curve, given as number of data points
        ext_x (float): Gaussian curve will extend from ext_x * sigma in each direction

    Returns:
        ndarray: filtered data

    """

    wind = _gauss(sigma, ext_x)
    wlen = len(wind)

    assert wlen <= len(x), "Window size must be smaller than data size"
    assert sigma * ext_x >= 1, "Window size must be larger than 1. Increase ext_x."

    s = np.r_[x[wlen - 1:0:-1], x, x[-2:-wlen - 1:-1]]
    y_out = np.convolve(wind / wind.sum(), s, mode='valid')
    y_out = y_out[wlen // 2:-wlen // 2 + 1]

    return y_out


def _gauss(sigma, n_sigma=3):
    """Generate a discrete, normalized Gaussian centered on zero.

    Used for filtering data.

    Args:
        sigma (float): standard deviation
        n_sigma (float): extend x in each direction by ext_x * sigma

    Returns:
        ndarray: discrete Gaussian curve

    """

    x_range = n_sigma * sigma
    x = np.arange(-x_range, x_range + 1e-5, 1, dtype=float)

    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / sigma)**2)

    return y


# MAIN -------------------------------------------------------------------- ##

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Create noisy data
    npts = 501
    yy = np.sin(np.linspace(0, 10, npts))
    yy_noise = yy + (np.random.random(npts) - 0.5) / 2

    # Filter noisy data
    yy_filter = gauss_conv(yy, 5)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(yy, 'b', label="Original Data")
    plt.plot(yy_noise, 'b', alpha=0.5, label="Noisy Data")
    plt.plot(yy_filter, 'r--', label="Filtered")
    plt.autoscale(axis='x', enable=True, tight=True)
    plt.legend()
    plt.show()
