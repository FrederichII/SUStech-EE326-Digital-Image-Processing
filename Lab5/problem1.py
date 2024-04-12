import numpy as np
import cv2
import fft
import sobel
import zeropadding

if __name__ == '__main__':
    img = cv2.imread('./Q5_1.tif',cv2.IMREAD_GRAYSCALE)
    print(img.dtype)
    sobel_sp = sobel.sobel(img)
    sobel_sp = fft.min_max(sobel_sp)
    f_pad = np.zeros([img.shape[0]+2,img.shape[1]+2],dtype = np.float32)
    f_pad[0:600,0:600] = img
    f_pad_u8 = np.uint8(f_pad)
    hx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
    print(hx.shape)
    hy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float32)
    hx_padded = np.pad(hx,((300,299),(300,299)),'constant',constant_values=0)
    hy_padded = np.pad(hy,((300,299),(300,299)), mode='constant',constant_values=0)
    for i in range(hx_padded.shape[0]):
        for j in range(hx_padded.shape[1]):
            if((i+j)%2 == 1):
                hx_padded[i,j] = -hx_padded[i,j]
                hy_padded[i,j] = -hy_padded[i,j]

    Hx = cv2.dft(hx_padded,flags=cv2.DFT_COMPLEX_OUTPUT)
    Hy = cv2.dft(hy_padded,flags=cv2.DFT_COMPLEX_OUTPUT)
    Hx_imag = Hx[:,:,1]
    Hy_imag = Hy[:,:,1]
    for i in range(Hx_imag.shape[0]):
        for j in range(Hx_imag.shape[1]):
            if((i+j)%2 == 1):
                Hx_imag[i,j] = -Hx_imag[i,j]
                Hy_imag[i,j] = -Hy_imag[i,j]

    Hx_u8 = np.uint8(cv2.normalize(Hx_imag, None, 0, 255, cv2.NORM_MINMAX))
    Hy_u8 = np.uint8(cv2.normalize(Hy_imag, None, 0, 255, cv2.NORM_MINMAX))
    F = np.fft.fft2(f_pad,axes=(-2,-1),norm=None)
    F = np.fft.fftshift(F)
    F_mag = cv2.magnitude(F.real,F.imag)
    F_mag, F_log = fft.spectrum(F_mag)
    F_u8 = np.uint8(cv2.normalize(F_log, None, 0, 255, cv2.NORM_MINMAX))
    Hx_imag = j*Hx_imag # imaginary number
    Hy_imag = j*Hy_imag # imaginary number
    F_x = F*Hx_imag
    F_y = F*Hy_imag
    F = F_x + F_y
    f_ifft = np.fft.ifft2(F,axes=(-2,-1),norm=None)
    f_ifft = np.sqrt(f_ifft.real**2+f_ifft.imag**2)
    f_ifft = np.log10(f_ifft)
    f_max = np.max(f_ifft)
    f_min = np.min(f_ifft)
    f_max_arg = np.argmax(f_ifft)
    f_min_arg = np.argmin(f_ifft)

    sobel_f = np.uint8(255 * (f_ifft - f_min) / (f_max - f_min))
    sobel_f = cv2.medianBlur(sobel_f,5)
    sobel_f = cv2.blur(sobel_f, (5,5))


    print(hx_padded.shape)
    print(hx_padded.dtype)


    cv2.imshow('spatial sobel', sobel_sp)
    cv2.imshow('f_pad',f_pad_u8)
    cv2.imshow('Hx_u8',Hx_u8)
    cv2.imshow('Hy_u8',Hy_u8)
    cv2.imshow('F_u8',F_u8)
    cv2.imshow('frequency sobel',sobel_f)
    cv2.waitKey(0)