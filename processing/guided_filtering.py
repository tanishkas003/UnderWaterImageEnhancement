import cv2
import numpy as np


def guided_filter(I, p, r, eps):
    """
    Guided filter implementation (grayscale-safe)
    """

    # 🔒 Ensure both inputs are single-channel
    if len(I.shape) == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    if len(p.shape) == 3:
        p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)

    I = I.astype(np.float64)
    p = p.astype(np.float64)

    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b

    return q

def guided_filter_enhancement(img):

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = v.astype(np.float64) / 255.0

    # Gamma correction
    gamma = 0.8
    corrected = np.power(v, gamma)

    # Guided filter
    transmission = guided_filter(corrected, corrected, r=15, eps=1e-3)

    t0 = 0.3
    transmission = np.maximum(transmission, t0)

    J = corrected / (transmission + 0.1)
    J = np.clip(J, 0, 1)

    # Normalize
    J = cv2.normalize(J, None, 0, 255, cv2.NORM_MINMAX)
    J = np.uint8(J)

    # Replace V channel
    hsv[:, :, 2] = J

    # Convert back to BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result