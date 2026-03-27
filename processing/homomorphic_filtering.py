# homomorphic_filtering.py

import cv2
import numpy as np

def homomorphic_filter(img):
    import cv2
    import numpy as np

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32) + 1

    log_img = np.log(img_gray)

    fft = np.fft.fft2(log_img)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    sigma = 30
    gamma_l = 0.5
    gamma_h = 1.5

    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    D2 = (X - ccol)**2 + (Y - crow)**2

    H = (gamma_h - gamma_l) * (1 - np.exp(-D2 / (2 * sigma**2))) + gamma_l

    filtered = fft_shift * H

    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    exp_img = np.exp(img_back)

    exp_img = cv2.normalize(exp_img, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(exp_img)
