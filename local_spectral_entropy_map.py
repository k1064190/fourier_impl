import cv2
import numpy as np
import scipy
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import view_as_blocks

def local_spectral_entropy_map(image, block_size=(8, 8)):
    if image.ndim == 3:
        image = rgb2gray(image)

    blocks = view_as_blocks(image, block_size)
    num_blocks = blocks.shape[0] * blocks.shape[1]
    entropy_map = np.zeros(image.shape)

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i, j]
            block_fft = fft2(block)
            mag = np.abs(block_fft)
            mag /= np.sum(mag)
            entropy = -np.sum(mag * np.log1p(mag))
            entropy_map[i* block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]] = entropy

    return entropy_map

image = cv2.imread('./khan.png', 0)
image = cv2.resize(image, (1024, 1024))
# image = scipy.misc.face()

entropy_map = local_spectral_entropy_map(image)

plt.imshow(entropy_map, cmap='gray')
plt.show()






