import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


laplacian = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

blur_kernel = np.array([1,3,3,1])
blur_kernel = blur_kernel.reshape(-1, 1) * blur_kernel.reshape(1, -1)
blur_kernel = blur_kernel / np.sum(blur_kernel)
print(blur_kernel)

fft_filter = np.fft.fft2(laplacian)
fft_shift = np.fft.fftshift(fft_filter)

mag_spect = np.log1p(np.abs(fft_shift))
# print(mag_spect)
# plt.imshow(mag_spect, cmap='gray')
# plt.show()

mag_spect = cv.resize(mag_spect, (512, 512))
# plt.imshow(mag_spect, cmap='gray')
# plt.show()

# plt.imshow(blur_kernel, cmap='gray')
# plt.show()
blur_fft = np.fft.fft2(blur_kernel)
blur_fft_shift = np.fft.fftshift(blur_fft)
blur_spect = np.log1p(np.abs(blur_fft_shift))
print(blur_spect)

# plt.imshow(blur_spect, cmap='gray')
# plt.show()

blur_spect = cv.resize(blur_spect, (512, 512))
plt.imshow(blur_spect, cmap='gray')
plt.show()

image = cv.imread('./Lenna.png', 0)

fft_image = np.fft.fft2(image)
fft_image_shift = np.fft.fftshift(fft_image)

producted1 = fft_image_shift * mag_spect
ifft_image = np.fft.ifft2(producted1)
ifft_image = np.abs(ifft_image)
ifft_image = np.log1p(ifft_image)
ifft_image1 = ifft_image

producted2 = fft_image_shift * blur_spect
producted2 = producted2 * blur_spect
producted2 = producted2 * blur_spect
ifft_image = np.fft.ifft2(producted2)
ifft_image = np.abs(ifft_image)
ifft_image = np.log1p(ifft_image)
ifft_image2 = ifft_image

# conv with blur kernel
blur_image = cv.filter2D(image, -1, blur_kernel)

plt.figure(figsize=(19, 19))
plt.subplot(141), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(ifft_image1, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(ifft_image2, cmap='gray')
plt.title('Blur Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(blur_image, cmap='gray')
plt.title('Blur Image'), plt.xticks([]), plt.yticks([])

plt.show()

