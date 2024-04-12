import histogram_equalization as hseq
import cv2
import matplotlib.pyplot as plt
import numpy as np


def specification(hist):
    spe_hist = np.zeros(hist.shape)
    seg1 = spe_hist[0:9,0]
    rate1 = 7 / len(seg1)
    for i in range(len(seg1)):
        seg1[i] = (i+1)*rate1

    seg2 = spe_hist[10:25,0]
    rate2 = (7-0.875) / len(seg2)
    for i in range(len(seg2)):
        seg2[i] = 7 - rate2*(i+1)

    seg3 = spe_hist[26:180,0]
    rate3 = 0.875/len(seg3)
    for i in range(len(seg3)):
        seg3[i] = 0.875 -  rate3*(i+1)

    seg4 = spe_hist[181:210,0]
    rate4 = 0.8/len(seg4)
    for i in range(len(seg4)):
        seg4[i] = rate4 * (i+1)

    seg5 = spe_hist[211:255,0]
    rate5 = 0.8/len(seg5)
    for i in range(len(seg5)):
        seg5[i] = 0.8 - rate5*(i+1)

    spe_hist[0:9,0] =seg1
    spe_hist[10:25,0] = seg2
    spe_hist[26:180,0] = seg3
    spe_hist[181:210,0] = seg4
    spe_hist[211:255,0] = seg5

    spe_hist /= np.sum(spe_hist)
    return spe_hist

def cdf(hist):
    cdf_hist = np.zeros(hist.shape)
    sum = 0
    for i in range(hist.shape[0]):
        sum += hist[i, 0]
        cdf_hist[i, 0] = sum


    return cdf_hist

def hist_match_12012724(input_img,input_hist, spe_hist):
    cdf_in = cdf(input_hist)
    cdf_out = cdf(spe_hist)
    plt.figure(3)
    plt.bar(range(256),cdf_in[:, 0])
    plt.figure(4)
    plt.bar(range(256),cdf_out[:, 0])
    plt.show()
    transform = np.zeros(256)
    output_img =np.zeros(input_img.shape,dtype=np.uint8)
    for i in range(256):
        j = 0
        while(np.abs(cdf_out[j, 0]-cdf_in[i, 0])>=0.1):
            j+=1
        transform[i] = j

    for i in range(output_img.shape[1]):
        for j in range(output_img.shape[0]):
            output_img[j, i] = transform[input_img[j ,i]]
    plt.figure(5)
    plt.bar(range(256),hseq.draw_hist(output_img)[:,0])
    plt.show()
    return output_img


if __name__ == '__main__':
    img = cv2.imread('./Q3_2.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = hseq.draw_hist(img)
    plt.figure(1)
    plt.bar(range(256), hist[:, 0])
    spe_hist = specification(hist)
    plt.figure(2)
    plt.bar(range(256),spe_hist[:, 0])
    plt.show()

    output_img = hist_match_12012724(img, hist, spe_hist)
    cv2.imshow('input image',cv2.resize(img,dsize=None,dst=None,fx=0.8,fy=0.8))
    cv2.imshow('output image',cv2.resize(output_img,dsize=None,dst=None,fx=0.8,fy=0.8))
    cv2.waitKey(0)




