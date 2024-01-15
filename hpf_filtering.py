import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./Lenna.png', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

magnitude_spectrum = 20*np.log(np.abs(fshift))
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

mask = np.ones((rows, cols), np.uint8)
mask[crow-20:crow+20, ccol-20:ccol+20] = 0

fshift = fshift * mask

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)

img_back = np.abs(img_back)

plt.figure(figsize=(19, 19))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
eps = 1e-10
magnitude_spectrum = np.clip(np.log(np.abs(fshift) + eps), 0, None)
plt.subplot(132), plt.imshow(np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum)) ,cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.show()

