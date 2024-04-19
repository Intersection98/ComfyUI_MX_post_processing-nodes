import cv2
import numpy as np
import math

# 2D LUT(size**3, size**3, 3) ->  3D LUT(size**3, size**3, 3)
clut = cv2.imread('1.png')
size = int(clut.shape[0] ** (1.0 / 3.0) + 0.5)
clut_result = np.zeros((size ** 2, size ** 2, size ** 2, 3))
for i in range(size ** 2):
    tmp1 = math.floor(i / size)
    cx = int((i - tmp1 * size) * size ** 2)
    cy = int(tmp1 * size ** 2)
    clut_result[i] = clut[cy: cy + size ** 2, cx : cx + size ** 2]

hald = clut_result.reshape((size ** 3, size ** 3, 3))

cv2.imwrite('2.png'.format(size), hald)

