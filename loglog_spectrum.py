import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./Lenna.png', cv2.IMREAD_GRAYSCALE)

fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

mags = 20 * np.log(np.abs(fft_shift))

row, col = mags.shape
center = (int(col/2), int(row/2))

# max_radius = int(np.sqrt(row ** 2 + col ** 2) / 2)
max_radius = center[0]

polar = cv2.linearPolar(mags, center, max_radius, cv2.WARP_FILL_OUTLIERS)

vprojec = np.mean(polar, axis=0)

plt.figure(figsize=(15, 15))
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.subplot(222), plt.imshow(mags, cmap='gray')
plt.subplot(223), plt.imshow(polar, cmap='jet', interpolation='bicubic')
plt.subplot(224), plt.plot(vprojec)
plt.yscale('log'), plt.xscale('log')
plt.show()

