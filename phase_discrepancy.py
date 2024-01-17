import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def complex_numpy(real, imag):
    return real * np.exp(1j * imag)

img = cv.imread('./khan.png', 0)
img = cv.resize(img, (256, 256))

# crop the image
diff = 64
print(img.shape)
img1 = img[0:128, 0:128]
img2 = img[diff:128+diff, diff:128+diff]

fft1 = np.fft.fft2(img1)
fft2 = np.fft.fft2(img2)

phase1 = np.angle(fft1)
phase2 = np.angle(fft2)

amp1 = np.abs(fft1)
amp2 = np.abs(fft2)

z1 = complex_numpy((amp1 - amp2), phase1)
z2 = complex_numpy((amp2 - amp1), phase2)

m1 = np.fft.ifft2(z1)
m2 = np.fft.ifft2(z2)

m11 = np.abs(m1)
m22 = np.abs(m2)

plt.figure(figsize=(30, 15))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
plt.show()

m12 = np.multiply(m11, m22)
result = np.interp(m12, (m12.min(), m12.max()), (0, 255))

plt.figure(figsize=(15, 15))
plt.imshow(result, cmap='gray')
plt.show()

