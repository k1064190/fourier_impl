import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('./khan.png', 0)
size = 32
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)
fft2 = np.fft.fft2(img, (128, 128))
new_img = np.fft.ifft2(fft2).real
plt.imshow(new_img, cmap='gray')
plt.show()
c_y, c_x = fft.shape[0] // 2, fft.shape[1] // 2
fft_low = fft_shift[c_y - size:c_y + size, c_x - size:c_x + size].copy()
mask = np.zeros(fft_shift.shape)
mask[c_y - size:c_y + size, c_x - size:c_x + size] = 1
fft_low2 = fft_shift.copy()
fft_low2[mask == 0] = 0
# low_img = np.fft.ifftshift(fft_low)
# low_img = np.fft.ifft2(low_img).real
low_img = np.fft.ifft2(fft2).real
low_img2 = np.fft.ifftshift(fft_low2)
low_img2 = np.fft.ifft2(low_img2, img.shape).real

fft_low3 = np.complex_(np.zeros(fft_shift.shape))
fft_low3[c_y - size:c_y + size, c_x - size:c_x + size] = fft_low
low_img3 = np.fft.ifftshift(fft_low3)
low_img3 = np.fft.ifft2(low_img3, img.shape).real


plt.figure(figsize=(15, 15))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 4, 2)
plt.imshow(low_img, cmap='gray')
plt.subplot(1, 4, 3)
plt.imshow(low_img2, cmap='gray')
plt.subplot(1, 4, 4)
plt.imshow(low_img3, cmap='gray')
plt.show()