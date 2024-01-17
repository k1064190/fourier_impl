import numpy as np
import cv2 as cv
import scipy
import matplotlib.pyplot as plt

gray = scipy.misc.face()
gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

window = np.hamming(gray.shape[0])[:, None] * np.hamming(gray.shape[1])[None, :]
gray_window = gray * window

fft = np.fft.fft2(gray)
fft_window = np.fft.fft2(gray_window)

fft_shift = np.fft.fftshift(fft)
fft_window_shift = np.fft.fftshift(fft_window)

mag = np.log1p(np.abs(fft_shift))
mag_window = np.log1p(np.abs(fft_window_shift))

plt.figure(figsize=(30, 30))
plt.subplot(221)
plt.imshow(gray, cmap='gray')
plt.subplot(222)
plt.imshow(gray_window, cmap='gray')
plt.subplot(223)
plt.imshow(mag, cmap='gray')
plt.subplot(224)
plt.imshow(mag_window, cmap='gray')
plt.show()

