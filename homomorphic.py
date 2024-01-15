import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./Lenna.png', 0)
rows, cols = img.shape

imgLog = np.log1p(np.array(img, dtype="float") / 255)

M = 2 * rows + 1
N = 2 * cols + 1

sigma = 10
(X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))

X_c = np.ceil(N/2)
Y_c = np.ceil(M/2)

gaussianNumerator = (X - X_c)**2 + (Y - Y_c)**2

LPF = np.exp(-gaussianNumerator / (2 * sigma * sigma))
HPF = 1 - LPF

LPF_shift = np.fft.ifftshift(LPF.copy())
HPF_shift = np.fft.ifftshift(HPF.copy())

# cv.imshow('LPF', np.uint8(LPF_shift / np.max(LPF_shift) * 255))
# cv.imshow('HPF', np.uint8(HPF_shift / np.max(HPF_shift) * 255))
# cv.waitKey(0)

img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N)))
img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N)))

gamma1 = 0.3
gamma2 = 1.5
img_adjusting = gamma1 * img_LF[0:rows, 0:cols] + gamma2 * img_HF[0:rows, 0:cols]

img_exp = np.expm1(img_adjusting)

img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) * 255
img_out = np.array(img_exp, dtype="uint8")

img_G = img_out
result = cv.cvtColor(img_G, cv.COLOR_GRAY2BGR)
cv.imshow('result', result)
cv.waitKey(0)
