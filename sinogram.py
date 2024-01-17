import numpy as np
import imutils
import skimage
import scipy.fftpack as fft
import scipy
import cv2 as cv
import matplotlib.pyplot as plt

def radon(image, steps):
    projections = []
    dTheta = -180.0 / steps
    for i in range(steps):
        rotated = imutils.rotate(image, i * dTheta)
        # cv.imshow('rotated', rotated)
        # cv.waitKey(0)
        projection = np.sum(rotated, axis=0)
        projections.append(projection)

    return projections

def fft_translate(projs):
    return fft.rfft(projs, axis=1)

def ramp_filter(ffts):
    ramp = np.floor(np.arange(0.5, ffts.shape[1] // 2 + 0.1, 0.5))
    return ffts * ramp

def inverse_fft_translate(operator):
    return fft.irfft(operator, axis=1)

def back_project(operator):
    laminogram = np.zeros((operator.shape[1], operator.shape[1]))
    dTheta = 180.0 / operator.shape[0]
    for i in range(operator.shape[0]):
        temp = np.tile(operator[i], (operator.shape[1], 1))
        temp = skimage.transform.rotate(temp, i * dTheta)
        laminogram += temp
    return laminogram

img = cv.imread('./khan.png', 0)
sinogram = radon(img, 180)
sinogram = np.array(sinogram, dtype=np.float32)

plt.imshow(sinogram, cmap='gray')
plt.title('sinogram')
plt.show()

unfiltered_reconstruction = back_project(sinogram)
plt.imshow(unfiltered_reconstruction, cmap='gray')
plt.title('unfiltered reconstruction')
plt.show()

frequency_domain_sinogram = fft_translate(sinogram)

plt.imshow(frequency_domain_sinogram.real, cmap='gray')
plt.title('frequency domain sinogram')
plt.show()

filtered_frequency_domain_sinogram = ramp_filter(frequency_domain_sinogram)
plt.imshow(filtered_frequency_domain_sinogram.real, cmap='gray')
plt.title('filtered frequency domain sinogram')
plt.show()

filtered_sinogram = inverse_fft_translate(filtered_frequency_domain_sinogram)
plt.imshow(filtered_sinogram.real, cmap='gray')
plt.title('filtered sinogram')
plt.show()

filtered_reconstruction = back_project(filtered_sinogram)
plt.imshow(filtered_reconstruction.real, cmap='gray')
plt.title('filtered reconstruction')
plt.show()

window = np.hamming(filtered_reconstruction.shape[0])
windowed = filtered_reconstruction * window
plt.imshow(windowed.real, cmap='gray')
plt.title('windowed image')
plt.show()



