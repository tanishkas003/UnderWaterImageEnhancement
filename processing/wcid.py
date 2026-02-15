import cv2
import numpy as np
import pywt

def wavelet_decompose(img):
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

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
    return transmission

def enhance_high_frequency(LH, HL, HH, transmission):
    threshold = np.mean(transmission) * 0.1
    LH = np.where(np.abs(LH) > threshold, LH * 1.2, LH)
    HL = np.where(np.abs(HL) > threshold, HL * 1.2, HL)
    HH = np.where(np.abs(HH) > threshold, HH * 1.2, HH)
    return LH, HL, HH

def wcid_enhancement(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255

    LL, LH, HL, HH = wavelet_decompose(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    dark = dark_channel(img)
    A = estimate_atmospheric_light(img, dark)
    transmission = estimate_transmission(img, A)

    LH, HL, HH = enhance_high_frequency(LH, HL, HH, transmission)

    reconstructed = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    reconstructed = np.clip(reconstructed, 0, 1)

    return (reconstructed * 255).astype(np.uint8)
