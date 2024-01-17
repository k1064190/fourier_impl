import numpy as np
import cv2 as cv
import scipy
import matplotlib.pyplot as plt

# contour1 = np.array([[110, 100], [200, 110], [210, 200], [90, 200]], dtype=np.float32)
# contour2 = np.array([[60, 50], [140, 50], [120, 150], [40, 150]], dtype=np.float32)
#
# plt.plot(contour1[:, 0], contour1[:, 1], 'r-', label='contour1')
# plt.plot(contour2[:, 0], contour2[:, 1], 'b-', label='contour2')
#
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
#
# plt.show()

def fourier_descriptor(contour):
    # contour to complex number
    comp = np.empty(contour.shape[0], dtype=np.complex64)
    comp.real = contour[:, 0]
    comp.imag = contour[:, 1]

    dft = np.fft.fft(comp)

    return dft

# fd1 = fourier_descriptor(contour1)
# fd2 = fourier_descriptor(contour2)
#
# print(fd1)
# print(fd2)
#
# difference = np.abs(fd1 - fd2)
# print(difference)
#
# mean_diff = np.mean(difference)
# print(mean_diff)

img = cv.imread('./khan.png', 0)
import phasepack
pc, _, _, _ = phasepack.phasecongmono(img)
pc = np.interp(pc, (pc.min(), pc.max()), (0, 255))
pc = np.array(pc, dtype=np.uint8)
# plt.imshow(pc, cmap='gray')
# plt.show()

contours, hierarchy = cv.findContours(pc, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# select contour which has the most points
contour = contours[0]
for c in contours:
    if len(c) > len(contour):
        contour = c

print(len(contour))
# draw contour
# cv.drawContours(img, contour, -1, (0, 255, 0), 3)
#
# plt.imshow(img)
# plt.show()

contour = np.squeeze(contour)
print(contour.shape)
fd = fourier_descriptor(contour)
print(fd.shape)

# 20개의 coefficient만 사용
fd[20:] = 0

# inverse fourier transform
idft = np.fft.ifft(fd)
x = idft.real
y = idft.imag
idft = np.empty((len(x), 2), dtype=np.float32)
idft[:, 0] = x
idft[:, 1] = y

# draw contour
contour = idft.astype(np.int32)
cv.drawContours(img, [contour], -1, (0, 255, 0), 3)

plt.imshow(img)
plt.show()



