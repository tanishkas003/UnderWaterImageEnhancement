# contrast_maximization.py

import numpy as np

def contrast_maximization(img):
    """
    Contrast Maximization / Enhancement
    Input: BGR image (numpy array)
    Output: Enhanced BGR image
    """

    img = img.astype(np.float32) / 255.0

    # Step 1: Determine size
    h, w, c = img.shape

    # Step 2: Calculate enhancement variable (k)
    mean_intensity = np.mean(img)
    k = 0.5 / (mean_intensity + 1e-6)

    # Step 3: Normalize using k
    enhanced = img * k
    enhanced = np.clip(enhanced, 0, 1)

    return (enhanced * 255).astype(np.uint8)
