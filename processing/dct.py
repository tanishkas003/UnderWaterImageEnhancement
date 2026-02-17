import cv2
import numpy as np
from scipy.fftpack import dct, idct


class DCT:
    def __init__(self, block_size=8, dc_boost=1.2):
        self.block_size = block_size
        self.dc_boost = dc_boost

    def block_dct(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def block_idct(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def apply(self, image):
        img = cv2.resize(image, (512, 512))

        # Step 2: DHE (Histogram Equalization on Y channel)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        # Step 3: Convert to YCbCr
        ycbcr = cv2.cvtColor(img_eq, cv2.COLOR_BGR2YCrCb)
        Y = ycbcr[:, :, 0].astype(np.float32)

        h, w = Y.shape
        Y_dct = np.zeros_like(Y)

        # Step 4–6: Block-wise DCT processing
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = Y[i:i+self.block_size, j:j+self.block_size]

                if block.shape == (self.block_size, self.block_size):
                    dct_block = self.block_dct(block)

                    # Boost DC component (local illumination enhancement)
                    dct_block[0, 0] *= self.dc_boost

                    Y_dct[i:i+self.block_size, j:j+self.block_size] = \
                        self.block_idct(dct_block)

        Y_dct = np.clip(Y_dct, 0, 255)

        # Step 7: Merge blocks
        ycbcr[:, :, 0] = Y_dct
        final_img = cv2.cvtColor(
            ycbcr.astype(np.uint8),
            cv2.COLOR_YCrCb2BGR
        )

        # Step 8: Median filtering (noise removal)
        final_img = cv2.medianBlur(final_img, 3)

        return final_img
