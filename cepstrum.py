import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def calc_image_cepstrum(img):
    # img_ft = np.fft.fft2(img)
    # img_ft_log = np.log(1 + np.abs(img_ft))
    # img_ft_log_ft = np.fft.fft2(img_ft_log)
    # img_cepstrum = np.real(np.fft.fftshift(np.fft.ifft2(img_ft_log_ft)))

    img_ft = np.fft.fft2(img)
    img_ft_log = np.log(1 + np.abs(img_ft))
    img_cepstrum = np.abs(np.fft.fftshift(np.fft.ifft2(img_ft_log)))
    img_cepstrum = np.interp(img_cepstrum, (img_cepstrum.min(), img_cepstrum.max()), (0, 255))

    return img_cepstrum

gray = cv.imread('./khan.png', 0)
gray = cv.resize(gray, (256, 256))
gray = gray

gray_cepstrum = calc_image_cepstrum(gray)

color = cv.imread('./test_img.png')
color = cv.resize(color, (256, 256))
c_gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
c_gray = c_gray / 255.0

c_gray_cepstrum = calc_image_cepstrum(c_gray)

plt.figure(figsize=(45, 15))
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(gray_cepstrum, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(c_gray_cepstrum, cmap='gray')
plt.show()


