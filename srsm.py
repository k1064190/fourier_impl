import cv2 as cv
import numpy as np

img = cv.imread('./detect609.jpg', 0)
img = cv.resize(img, (256, 256))

fft = np.fft.fft2(img)
fftshift = np.fft.fftshift(fft)
magnitude_spectrum = np.log(np.abs(fftshift)+1)
phase = np.angle(fftshift)

# smooth the magnitude spectrum with a average filter
smooth = cv.blur(magnitude_spectrum, (3, 3))
sr = magnitude_spectrum - smooth

cv.imshow('sr', sr)
cv.waitKey(0)

saliencyMap = np.abs(np.fft.ifft2(np.exp(sr + 1j * phase))) ** 2
saliencyMap = cv.GaussianBlur(saliencyMap, (9, 9), 2.5)

saliencyMap = (saliencyMap - np.min(saliencyMap)) / (np.max(saliencyMap) - np.min(saliencyMap))
saliencyMap = np.uint8(saliencyMap * 255)

cv.imshow('saliencyMap', saliencyMap)
cv.waitKey(0)

