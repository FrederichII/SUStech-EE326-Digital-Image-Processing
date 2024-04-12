import numpy as np
import cv2


def FFT(img):
    img_float = img.astype(np.float32)
    fft = np.fft.fft2(img_float,axes=(-2,-1),norm=None)
    fft = np.fft.fftshift(fft)

    return fft


def min_max(x):
    return np.uint8((x-np.min(x))/(np.max(x)-np.min(x)) * 255)


def IFFT(img_fft):
    img_fft = np.fft.ifftshift(img_fft)
    img = np.fft.ifft2(img_fft,axes=(-2,-1),norm=None)
    img = np.sqrt(img.real**2 + img.imag**2)
    img = np.uint8(min_max(img) * 255)
    return img


def spectrum(img_fft):
    img_fft_mag = np.sqrt(np.power(img_fft.real, 2) + np.power(img_fft.imag, 2))
    img_fft_log = np.log(img_fft_mag)
    #img_fft_mag = np.uint8(min_max(img_fft_mag) * 255)
    #img_fft_log = np.uint8(min_max(img_fft_log) * 255)
    return img_fft_mag, img_fft_log


if __name__ == '__main__':
    img = cv2.imread('./Q5_1.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fft = FFT(img)
    # img_fft_mag = np.sqrt(np.power(img_fft.real,2)+np.power(img_fft.imag,2))
    # img_fft_log = np.log(img_fft_mag)
    # img_fft_mag = np.uint8(min_max(img_fft_mag) * 255)
    # img_fft_log = np.uint8(min_max(img_fft_log) * 255)
    img_fft_mag = spectrum(img_fft)[0]
    img_fft_log = spectrum(img_fft)[1]
    cv2.imshow('img',img)
    cv2.imshow('magnitude spectrum',img_fft_mag)
    cv2.imshow('log spectrum',img_fft_log)
    cv2.waitKey(0)
