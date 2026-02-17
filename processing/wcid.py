import cv2
import numpy as np
import pywt


class WCID:
    def __init__(self, patch=15, omega=0.95):
        self.patch = patch
        self.omega = omega

    def wavelet_decompose(self, img):
        coeffs2 = pywt.dwt2(img, 'haar')
        LL, (LH, HL, HH) = coeffs2
        return LL, LH, HL, HH

    def dark_channel(self, img):
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch, self.patch)
        )
        dark = cv2.erode(min_channel, kernel)
        return dark

    def estimate_atmospheric_light(self, img, dark):
        flat_dark = dark.ravel()
        flat_img = img.reshape(-1, 3)

        num_pixels = int(0.001 * len(flat_dark))
        indices = np.argsort(flat_dark)[-num_pixels:]
        A = np.mean(flat_img[indices], axis=0)
        return A

    def estimate_transmission(self, img, A):
        norm_img = img / (A + 1e-6)
        transmission = 1 - self.omega * self.dark_channel(norm_img)
        return transmission

    def enhance_high_frequency(self, LH, HL, HH, transmission):
        threshold = np.mean(transmission) * 0.1

        LH = np.where(np.abs(LH) > threshold, LH * 1.2, LH)
        HL = np.where(np.abs(HL) > threshold, HL * 1.2, HL)
        HH = np.where(np.abs(HH) > threshold, HH * 1.2, HH)

        return LH, HL, HH

    def apply(self, image):
        img = cv2.resize(image, (512, 512))
        img = img.astype(np.float32) / 255.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        LL, LH, HL, HH = self.wavelet_decompose(gray)

        dark = self.dark_channel(img)
        A = self.estimate_atmospheric_light(img, dark)
        transmission = self.estimate_transmission(img, A)

        LH, HL, HH = self.enhance_high_frequency(LH, HL, HH, transmission)

        reconstructed = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
        reconstructed = np.clip(reconstructed, 0, 1)

        result = (reconstructed * 255).astype(np.uint8)

        # Convert grayscale back to 3-channel
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result
