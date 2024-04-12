import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    normalized_hist = hist / img.shape[0] / img.shape[1]
    return normalized_hist


def hist_equ_12012724(input_img):
    input_hist = cv2.calcHist([input_img], [0], None, [256], [0, 255])
    normalized_hist = input_hist / input_img.shape[0] / input_img.shape[1]
    transform = np.zeros(input_hist.shape)

    sum = 0
    for i in range(256):
        sum += normalized_hist[i]
        transform[i] = np.around(sum * 255)

    output_img = np.zeros(input_img.shape, dtype=np.uint8)
    for i in range(input_img.shape[1]):
        for j in range(input_img.shape[0]):
            output_img[j, i] = transform[input_img[j, i]]

    output_hist = draw_hist(output_img)
    return output_img, output_hist, input_hist


if __name__ == '__main__':
    img1 = cv2.imread('./Q3_1_1.tif')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    norm_hist1 = draw_hist(img1)
    output_img1, output_hist1, input_hist1 = hist_equ_12012724(img1)

    img2 = cv2.imread('./Q3_1_2.tif')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    norm_hist2 = draw_hist(img2)
    output_img2, output_hist2, input_hist2 = hist_equ_12012724(img2)

    plt.figure(1)
    plt.bar(range(256), norm_hist1[:, 0])
    plt.figure(2)
    plt.bar(range(256), output_hist1[:, 0])
    plt.figure(3)
    plt.bar(range(256), norm_hist2[:, 0])
    plt.figure(4)
    plt.bar(range(256), output_hist2[:, 0])
    plt.show()
    cv2.imshow('input image1', img1)
    cv2.imshow('output image1', output_img1)
    cv2.imshow('input image2', img2)
    cv2.imshow('output image2',output_img2)
    cv2.waitKey()
