import cv2
import numpy as np
from scipy.fftpack import dct, idct

def block_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def block_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def dct_enhancement(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    ycbcr = cv2.cvtColor(img_eq, cv2.COLOR_BGR2YCrCb)
    Y = ycbcr[:, :, 0].astype(np.float32)

    h, w = Y.shape
    block_size = 8
    Y_dct = np.zeros_like(Y)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = Y[i:i+block_size, j:j+block_size]
            if block.shape == (8, 8):
                dct_block = block_dct(block)

                dct_block[0, 0] *= 1.2

                Y_dct[i:i+block_size, j:j+block_size] = block_idct(dct_block)

    Y_dct = np.clip(Y_dct, 0, 255)

    ycbcr[:, :, 0] = Y_dct
    final_img = cv2.cvtColor(ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    final_img = cv2.medianBlur(final_img, 3)

    return final_img
