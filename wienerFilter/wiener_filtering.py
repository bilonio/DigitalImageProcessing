import numpy as np
import scipy.fftpack


def my_wiener_filter(y: np.ndarray, h: np.ndarray, K: float):
    fft_y = scipy.fftpack.fft2(y)  # 2D FFT of the distorted image
    fft_h = scipy.fftpack.fft2(
        h, shape=y.shape
    )  # 2D FFT of the impulse reponse of distortion system
    fft_y = np.array(fft_y)  # 2D FFT of the distortion as numpy array
    fft_h = np.array(fft_h)  # 2D FFT of the impulse response as numpy array
    fft_g = np.conj(fft_h) / (np.abs(fft_h) ** 2 + 1 / K)  # 2D FFT of the Wiener filter
    fft_x_hat = fft_y * fft_g  # 2D FFT of the estimated image
    x_hat = np.abs(scipy.fftpack.ifft2(fft_x_hat))  # Estimated image as numpy array
    return x_hat
