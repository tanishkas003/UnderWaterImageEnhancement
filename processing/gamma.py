import numpy as np
import cv2

class GammaCorrection:
    def __init__(self, gamma=1.2):
        self.gamma = max(gamma, 0.1)

    def apply(self, image):
        invGamma = 1.0 / self.gamma
        table = np.array([(i / 255.0) ** invGamma * 255
                         for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
