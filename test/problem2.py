import numpy as np
import cv2

img = cv2.imread('./Figure2.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img,(np.uint8(.8*img.shape[0]),np.uint8(.8*img.shape[1])))


cv2.imshow('origin',img)

c = 1
gammas = [0.25, 0.5, 0.75, 0.8,0.9]
title = []
for i in range(len(gammas)):
    img_gamma = c * np.power(img, gammas[i])
    img_gamma =np.uint8(img_gamma)
    img_gamma = cv2.equalizeHist(img_gamma)
    title = "gamma transform when exp = "+ str(gammas[i])
    cv2.imshow(title,img_gamma)

cv2.waitKey(0)