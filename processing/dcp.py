import cv2
import numpy as np

def dark_channel(img, patch=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(img, dark):
    flat_dark = dark.ravel()
    flat_img = img.reshape(-1, 3)

    indices = np.argsort(flat_dark)[-int(0.001 * len(flat_dark)):]
    A = np.mean(flat_img[indices], axis=0)
    return A


def estimate_transmission(img, A, patch=15, omega=0.95):
    norm_img = img / A
    transmission = 1 - omega * dark_channel(norm_img, patch)
    return np.clip(transmission, 0.1, 1)


def refine_transmission(img, transmission):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    refined = cv2.bilateralFilter(transmission.astype(np.float32), 9, 75, 75)
    return refined


def recover_scene_radiance(img, transmission, A):
    t = np.expand_dims(transmission, axis=2)
    J = (img - A) / t + A
    return np.clip(J, 0, 1)


def dcp_enhancement(image_path, patch_sizes=[7, 15, 25]):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255

    best_result = None
    best_contrast = 0

    for patch in patch_sizes:
        dark = dark_channel(img, patch)
        A = estimate_atmospheric_light(img, dark)
        transmission = estimate_transmission(img, A, patch)
        transmission = refine_transmission(img, transmission)
        recovered = recover_scene_radiance(img, transmission, A)

        contrast = np.std(recovered)
        if contrast > best_contrast:
            best_contrast = contrast
            best_result = recovered

    return (best_result * 255).astype(np.uint8)
