# histogram_equalization.py

import cv2

def histogram_equalization(img):
    """
    Histogram Equalization using Y channel (NTSC-like)
    Input: BGR image
    Output: Enhanced BGR image
    """

    # Convert BGR to YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Equalize Y channel
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    # Convert back
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return enhanced
