import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def get_magnitude_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))
    return magnitude_spectrum

def get_phase_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    phase_spectrum = np.angle(fshift)
    return phase_spectrum

def get_ifft2(img):
    f = np.fft.ifft2(img)
    magnitude_spectrum = np.log(np.abs(f))
    return magnitude_spectrum

def main():
    img = cv.imread('./Lenna.png', 0)
    img = cv.resize(img, (256, 256))
    ms = get_magnitude_spectrum(img)
    ps = get_phase_spectrum(img)

    plt.figure(figsize=(19, 19))
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(ms, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(ps, cmap='gray')
    plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

    img2 = cv.imread('./cman.jpg', 0)
    ms2 = get_magnitude_spectrum(img2)
    ps2 = get_phase_spectrum(img2)

    plt.figure(figsize=(19, 19))
    plt.subplot(131), plt.imshow(img2, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(ms2, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(ps2, cmap='gray')
    plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Swap phase spectrum
    fshift1 = ms * np.exp(1j * ps2)
    fshift2 = ms2 * np.exp(1j * ps)

    mag1 = get_ifft2(fshift1)
    mag2 = get_ifft2(fshift2)

    plt.figure(figsize=(19, 19))
    plt.subplot(121), plt.imshow(mag1, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(mag2, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()

