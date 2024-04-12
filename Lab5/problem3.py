import numpy as np
import cv2
import fft


def spectrum(img_fft):
    img_fft_mag = np.sqrt(np.power(img_fft.real, 2) + np.power(img_fft.imag, 2))
    img_fft_log = np.log(img_fft_mag)
    img_fft_mag = np.uint8(fft.min_max(img_fft_mag) * 255)
    img_fft_log = np.uint8(fft.min_max(img_fft_log) * 255)
    return img_fft_mag, img_fft_log


def Gaussian_lpf(height, width, d0):
    filter = np.zeros((height, width), dtype=np.float32)
    center = (width/2, height/2)
    for i in range(height):
        for j in range(width):
            dist2 = (i-center[0])**2 + (j-center[1])**2
            filter[i,j] = np.exp(-dist2**2/(2*d0**2))

    return filter

def Gaussian_hpf(height, width, d0):
    ones = np.ones((height, width), dtype=np.float32)
    filter = ones - Gaussian_lpf(height, width, d0)
    return filter

def Gaussian_lpf_filtering(img,d0):
    height, width = img.shape[0], img.shape[1]
    filter = Gaussian_lpf(height, width, d0)
    test = 255 * np.ones(img.shape, dtype=np.float32)
    test = test * filter
    test = np.uint8(255*(test - np.min(test))/(np.max(test) - np.min(test)))
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_filtered = filter * img_fft
    img_ifft = np.fft.ifft2(img_filtered)
    img_new = cv2.magnitude(img_ifft.real, img_ifft.imag)
    img_new = np.uint8(255*(img_new - np.min(img_new))/(np.max(img_new) - np.min(img_new)))
    title_fil = "Gaussian LPF with d0 = "+ str(d0)
    title_im = "Gaussian LPF image with d0 = " + str(d0)
    return title_im, img_new, title_fil, test


def Gaussian_hpf_filtering(img,d0):
    height, width = img.shape[0], img.shape[1]
    filter = Gaussian_hpf(height, width, d0)
    test = 255 * np.ones(img.shape, dtype=np.float32)
    test = test * filter
    test = np.uint8(255*(test - np.min(test))/(np.max(test) - np.min(test)))
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_filtered = filter * img_fft
    img_ifft = np.fft.ifft2(img_filtered)
    img_new = cv2.magnitude(img_ifft.real, img_ifft.imag)
    img_new = np.uint8(255*(img_new - np.min(img_new))/(np.max(img_new) - np.min(img_new)))
    title_fil = "Gaussian HPF with d0 = " + str(d0)
    title_im = "Gaussian HPF image with d0 = " + str(d0)
    return title_im, img_new, title_fil, test

if __name__ == '__main__':
    img = cv2.imread('./Q5_2.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fft = np.fft.fft2(img, axes=(-2, -1), norm=None)
    img_fft = np.fft.fftshift(img_fft)
    img_fft = cv2.magnitude(img_fft.real, img_fft.imag)
    img_fft = np.log(img_fft)
    img_fft = np.uint8(255*(img_fft - np.min(img_fft))/(np.max(img_fft) - np.min(img_fft)))

    cv2.imshow('img', img)
    cv2.imshow('img_fft', img_fft)
    title_im30l, img_new30l, title_fil30l, test30l = Gaussian_lpf_filtering(img,30)
    title_im30h, img_new30h, title_fil30h, test30h = Gaussian_hpf_filtering(img,30)

    title_im60l, img_new60l, title_fil60l, test60l = Gaussian_lpf_filtering(img,60)
    title_im60h, img_new60h, title_fil60h, test60h = Gaussian_hpf_filtering(img,60)

    title_im160l, img_new160l, title_fil160l, test160l = Gaussian_lpf_filtering(img,460)
    title_im160h, img_new160h, title_fil160h, test160h = Gaussian_hpf_filtering(img,460)

    cv2.imshow(title_fil30l, test30l)
    cv2.imshow(title_im30l, img_new30l)

    cv2.imshow(title_fil30h, test30h)
    cv2.imshow(title_im30h, img_new30h)

    cv2.imshow(title_fil60l, test60l)
    cv2.imshow(title_im60l, img_new60l)

    cv2.imshow(title_fil60h, test60h)
    cv2.imshow(title_im60h, img_new60h)

    cv2.imshow(title_fil160l, test160l)
    cv2.imshow(title_im160l, img_new160l)

    cv2.imshow(title_fil160h, test160h)
    cv2.imshow(title_im160h, img_new160h)

    cv2.waitKey(0)