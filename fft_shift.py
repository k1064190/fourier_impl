import numpy as np
from scipy import fftpack
import scipy
import matplotlib.pyplot as plt

gray = scipy.misc.ascent()

img = fftpack.fft2(gray)

dx = 150
dy = 90

fx, fy = np.meshgrid(np.arange(-gray.shape[0]//2, gray.shape[0]//2),
                     np.arange(-gray.shape[1]//2, gray.shape[1]//2))

img = img * np.exp(-2j * np.pi * dy * fx / gray.shape[1])
img = img * np.exp(-2j * np.pi * dx * fy / gray.shape[0])

plt.figure(figsize=(30, 15))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(np.abs(fftpack.ifft2(img)), cmap='gray')
plt.show()

