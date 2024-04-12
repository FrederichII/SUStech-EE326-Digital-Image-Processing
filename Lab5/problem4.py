import numpy as np
import cv2
import fft

def notch_filter(height, width, n, center, d0):
    notch = np.zeros((height, width), dtype=np.float32)
    M = height
    N = width
    for i in range(notch.shape[0]):
        for j in range(notch.shape[1]):
            notch[i,j] = 1
            for k in range(len(center)):
                uk = center[k][0]
                vk = center[k][1]
                dist1 = np.sqrt((i-uk)**2+(j-vk)**2)
                dist2 = np.sqrt((i-M+uk)**2+(j-N+vk)**2)
                notch[i,j] *= 1/(1+np.power((d0[k]/dist1),n))
                notch[i,j] *= 1/(1+np.power((d0[k]/dist2),n))
    return notch


if __name__ == '__main__':
    img = cv2.imread('Q5_3.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fft = np.fft.fft2(img)
    img_fft_u8 = cv2.magnitude(img_fft.real, img_fft.imag)
    img_fft_u8 = np.log(img_fft_u8)
    img_fft_u8 = fft.min_max(img_fft_u8)


    center = [[38,30],[205,28],[80,29],[164,27]]
    d0 = [9,9,4,4]
    n = 4
    height, width = img.shape
    notch = notch_filter(height, width, n, center, d0)
    test = 255*np.ones(img.shape, dtype=np.float32)
    test = test * notch
    test = fft.min_max(test)

    img_filtered = notch * img_fft
    img_filtered_u8 = np.sqrt(img_filtered.real**2 + img_filtered.imag**2)
    img_filtered_u8 = np.uint8( 255* (img_filtered_u8 - np.min(img_filtered_u8))/(np.max(img_filtered_u8)-np.min(img_filtered_u8)))
    img_new = np.fft.ifft2(img_filtered)
    img_new = np.sqrt(img_new.real**2 + img_new.imag**2)
    img_new = np.uint8( 255* (img_new - np.min(img_new))/(np.max(img_new)-np.min(img_new)))

    cv2.imshow('Original', img)
    cv2.imshow('img_fft',img_fft_u8)
    cv2.imshow('notch',test)
    cv2.imshow('img_filtered_u8',img_filtered_u8)
    cv2.imshow('result image',img_new)
    cv2.waitKey(0)