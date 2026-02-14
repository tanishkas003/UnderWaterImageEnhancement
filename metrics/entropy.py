import cv2
import numpy as np

def calculate_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    logs = np.log2(hist_norm + 1e-7)
    entropy = -1 * (hist_norm * logs).sum()
    return entropy
