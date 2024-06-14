import numpy as np
import scipy.fftpack


def inverse_filter(y, h, K):
    fft_h = np.array(scipy.fftpack.fft2(h, shape=y.shape))
    fft_y = np.array(scipy.fftpack.fft2(y))
    inv_H = 1 / fft_h
    x_hat = np.abs(scipy.fftpack.ifft2(fft_y * inv_H))
    return x_hat
