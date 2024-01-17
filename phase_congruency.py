import numpy as np
import phasepack
import matplotlib.pyplot as plt
import cv2 as cv


img = cv.imread('./khan.png', 0)
img_arr = np.asarray(img, dtype=np.float32)

pc, _, _, _ = phasepack.phasecongmono(img_arr)

plt.figure(figsize=(15, 15))
plt.imshow(pc, cmap='gray')
plt.show()




