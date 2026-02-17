# guided_filtering.py

import cv2
import numpy as np

def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I*p, cv2.CV_64F, (r, r))

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I*I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q


def guided_filter_enhancement(img):
    """
    Guided Filtering Based Enhancement
    Input: BGR image
    Output: Enhanced grayscale image
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    # Step 1: Artificial light correction (gamma)
    gamma = 0.8
    corrected = np.power(gray, gamma)

    # Step 2 & 3: Transmission map estimation + optimization
    transmission = guided_filter(corrected, corrected, r=15, eps=1e-3)

    # Step 4: Recover scene radiance
    J = corrected / (transmission + 1e-6)
    J = np.clip(J, 0, 1)

    return (J * 255).astype(np.uint8)
