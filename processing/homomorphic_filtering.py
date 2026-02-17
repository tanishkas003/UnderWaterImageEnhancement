# homomorphic_filtering.py

import cv2
import numpy as np

def homomorphic_filter(img):
    """
    Homomorphic Filtering (Frequency Domain)
    Input: BGR image
    Output: Enhanced grayscale image
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32) + 1

    # Log transform
    log_img = np.log(img_gray)

    # FFT
    fft = np.fft.fft2(log_img)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    sigma = 30
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    D2 = (X - ccol)**2 + (Y - crow)**2
    H = 1 - np.exp(-D2 / (2 * sigma**2))

    # Apply high-pass filter
    filtered = fft_shift * H

    # Inverse FFT
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    # Exponential
    exp_img = np.exp(img_back)
    exp_img = np.uint8(np.clip(exp_img, 0, 255))

    return exp_img
