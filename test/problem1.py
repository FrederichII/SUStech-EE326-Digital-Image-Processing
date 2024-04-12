import numpy as np
import cv2

img = cv2.imread('./Figure1.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img)

img_log = np.log(1.0+img)
img_log = np.uint8(255 * (img_log - np.min(img_log))/(np.max(img_log)-np.min(img_log)))
cv2.imshow('img_log', img_log)

cv2.waitKey(0)
