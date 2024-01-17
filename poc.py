import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def phase_correlation(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a * conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    return r

img = cv.imread('./khan.png', 0)

diff = 32
img1 = img[0:128, 0:128]
img2 = img[diff:128+diff, diff:128+diff]

result = phase_correlation(img1, img2)

plt.figure(figsize=(15, 15))
plt.imshow(result, cmap='gray')
plt.show()
