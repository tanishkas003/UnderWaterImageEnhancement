import cv2
import numpy as np


class DCP:
    def __init__(self, patch_sizes=[7, 15, 25], omega=0.95):
        self.patch_sizes = patch_sizes
        self.omega = omega

    def dark_channel(self, img, patch):
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (patch, patch)
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

    def estimate_transmission(self, img, A, patch):
        norm_img = img / (A + 1e-6)
        transmission = 1 - self.omega * self.dark_channel(norm_img, patch)
        return np.clip(transmission, 0.1, 1)

    def refine_transmission(self, transmission):
        refined = cv2.bilateralFilter(
            transmission.astype(np.float32), 9, 75, 75
        )
        return refined

    def recover_scene_radiance(self, img, transmission, A):
        t = np.expand_dims(transmission, axis=2)
        J = (img - A) / (t + 1e-6) + A
        return np.clip(J, 0, 1)

    def apply(self, image):
        img = cv2.resize(image, (512, 512))
        img = img.astype(np.float32) / 255.0

        best_result = None
        best_contrast = 0

        for patch in self.patch_sizes:
            dark = self.dark_channel(img, patch)
            A = self.estimate_atmospheric_light(img, dark)
            transmission = self.estimate_transmission(img, A, patch)
            transmission = self.refine_transmission(transmission)
            recovered = self.recover_scene_radiance(img, transmission, A)

            contrast = np.std(recovered)
            if contrast > best_contrast:
                best_contrast = contrast
                best_result = recovered

        result = (best_result * 255).astype(np.uint8)
        return result
