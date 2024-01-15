import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./churchill.jpg', 0)

def complex_numpy(real, imag):
    return real * np.exp(1j * imag)

img = cv.imread('./khan.png', 0)
img = cv.resize(img, (256, 256))

# crop the image
diff = 32
img1 = img[0:128, 0:128]
img2 = img[diff:128+diff, diff:128+diff]

fft1 = np.fft.fft2(img1)
fft2 = np.fft.fft2(img2)





