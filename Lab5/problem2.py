import numpy as np
import cv2
import fft


def spectrum(img_fft):
    img_fft_mag = np.sqrt(np.power(img_fft.real, 2) + np.power(img_fft.imag, 2))
    img_fft_log = np.log(img_fft_mag)
    img_fft_mag = np.uint8(fft.min_max(img_fft_mag) * 255)
    img_fft_log = np.uint8(fft.min_max(img_fft_log) * 255)
    return img_fft_mag, img_fft_log


def ILPF(height, width, d0):
    filter = np.zeros((height, width), dtype=np.float32)
    center = [height / 2 ,width / 2]
    for i in range(height):
        for j in range(width):
            if (np.sqrt((i-center[0])**2 + (j-center[1])**2) < d0):
                filter[i, j] = 1
    return filter


def blur_with_ILPF(img,height, width, d0):
    filter = ILPF(height, width, d0)

    # filter the image
    img_fft = np.fft.fft2(img, axes=(-2, -1), norm=None)
    img_fft = np.fft.fftshift(img_fft)
    img_ILPF = filter * img_fft
    img_ifft = np.fft.ifft2(img_ILPF)
    img_new = cv2.magnitude(img_ifft.real, img_ifft.imag)
    img_new = fft.min_max(img_new)
    title = "ILPF-filtered image with D0 = "+ str(d0)
    return title, img_new



img = cv2.imread('./Q5_2.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_fft = np.fft.fft2(img,axes=(-2,-1),norm=None)
img_fft = np.fft.fftshift(img_fft)
img_fft_mag, img_fft_log = spectrum(img_fft)

cv2.imshow('img',img)
cv2.imshow('img_fft_mag',img_fft_mag)
cv2.imshow('img_fft_log',img_fft_log)

title1, img_new1 = blur_with_ILPF(img,img.shape[0], img.shape[1], 10)
title2, img_new2 = blur_with_ILPF(img,img.shape[0], img.shape[1], 30)
title3, img_new3 = blur_with_ILPF(img,img.shape[0], img.shape[1], 60)
title4, img_new4 = blur_with_ILPF(img,img.shape[0], img.shape[1], 160)
title5, img_new5 = blur_with_ILPF(img,img.shape[0], img.shape[1], 460)

cv2.imshow(title1,img_new1)
cv2.imshow(title2,img_new2)
cv2.imshow(title3,img_new3)
cv2.imshow(title4,img_new4)
cv2.imshow(title5,img_new5)

cv2.waitKey(0)

