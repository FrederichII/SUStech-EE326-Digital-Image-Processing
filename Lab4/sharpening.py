import cv2
import numpy as np
import median_filter
import averaging_filter


def laplacian(img):

    mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    # enlarging the img
    enlarged_img = np.zeros([img.shape[0]+2,img.shape[1]+2])
    enlarged_img[1:enlarged_img.shape[0]-1,1:enlarged_img.shape[1]-1] = img
    enlarged_img[0,1:enlarged_img.shape[1]-1] = img[0,:]
    enlarged_img[enlarged_img.shape[0]-1,1:enlarged_img.shape[1]-1] = img[img.shape[0]-1,:]
    enlarged_img[1:enlarged_img.shape[0]-1,0] = img[:,0]
    enlarged_img[1:enlarged_img.shape[0]-1,enlarged_img.shape[1]-1] = img[:,img.shape[1]-1]

    enlarged_img[0,0] = img[0,0]
    enlarged_img[0,enlarged_img.shape[1]-1] = img[0,img.shape[1]-1]
    enlarged_img[enlarged_img.shape[0]-1,0] = img[img.shape[0]-1,0]
    enlarged_img[enlarged_img.shape[0]-1,enlarged_img.shape[1]-1] = img[img.shape[0]-1,img.shape[1]-1]
    # conducting Laplacian masking
    output_img = np.zeros(enlarged_img.shape, dtype=np.uint8)
    for i in range(1,enlarged_img.shape[0]-1):
        for j in range(1,enlarged_img.shape[1]-1):
            tmp = enlarged_img
            frame = np.array([[tmp[i-1,j-1],tmp[i-1,j],tmp[i-1,j+1]],[tmp[i,j-1],tmp[i,j],tmp[i,j+1]],[tmp[i+1,j-1],tmp[i+1,j],tmp[i+1,j+1]]])
            output_img[i,j] = np.sum(frame * mask)
    # slicing to get the result
    result = output_img[1:enlarged_img.shape[0]-1,1:enlarged_img.shape[1]-1]
    return result
def sobel(img):
    mask1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    mask2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    # enlarging the img
    enlarged_img = np.zeros([img.shape[0] + 2, img.shape[1] + 2])
    enlarged_img[1:enlarged_img.shape[0] - 1, 1:enlarged_img.shape[1] - 1] = img
    enlarged_img[0, 1:enlarged_img.shape[1] - 1] = img[0, :]
    enlarged_img[enlarged_img.shape[0] - 1, 1:enlarged_img.shape[1] - 1] = img[img.shape[0] - 1, :]
    enlarged_img[1:enlarged_img.shape[0] - 1, 0] = img[:, 0]
    enlarged_img[1:enlarged_img.shape[0] - 1, enlarged_img.shape[1] - 1] = img[:, img.shape[1] - 1]

    enlarged_img[0, 0] = img[0, 0]
    enlarged_img[0, enlarged_img.shape[1] - 1] = img[0, img.shape[1] - 1]
    enlarged_img[enlarged_img.shape[0] - 1, 0] = img[img.shape[0] - 1, 0]
    enlarged_img[enlarged_img.shape[0] - 1, enlarged_img.shape[1] - 1] = img[img.shape[0] - 1, img.shape[1] - 1]
    # gradient
    output_img = np.zeros(enlarged_img.shape, dtype=np.uint8)
    for i in range(1, enlarged_img.shape[0] - 1):
        for j in range(1, enlarged_img.shape[1] - 1):
            tmp = enlarged_img
            frame = np.array([[tmp[i-1,j-1],tmp[i-1,j],tmp[i-1,j+1]],[tmp[i,j-1],tmp[i,j],tmp[i,j+1]],[tmp[i+1,j-1],tmp[i+1,j],tmp[i+1,j+1]]])
            output_img[i,j] = np.abs(np.sum(frame * mask1)) + np.abs(np.sum(frame * mask2))

    # slicing to get the result
    result = output_img[1:enlarged_img.shape[0] - 1,1:enlarged_img.shape[1]-1]
    return result

def hist_match(img,ref):
        histImg, bins = np.histogram(img, 256)
        histRef, bins = np.histogram(ref, 256)
        cdfImg = histImg.cumsum()
        cdfRef = histRef.cumsum()

        transM = np.zeros(256)
        for i in range(256):
            index = 0
            vMin = np.fabs(cdfImg[i] - cdfRef[0])
            for j in range(256):
                diff = np.fabs(cdfImg[i] - cdfRef[j])
                if (diff < vMin):
                    index = int(j)
                    vMin = diff
            transM[i] = index


        img = transM[img].astype(np.uint8)
        return img

def sharpening(img):
    img_laplacian_raw = laplacian(img)
    img_laplacian = img_laplacian_raw + img
    img_laplacian = median_filter.median(7,img_laplacian)
    img_sobel_raw = sobel(img)
    output = np.zeros(img.shape,dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = 0.8*img_laplacian[i,j] + 0.2*img_sobel_raw[i,j]
    output = hist_match(output, img)
    return output


if __name__ == '__main__':
    img1 = cv2.imread('Q4_1.tif')
    img2 = cv2.imread('Q4_2.tif')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = averaging_filter.averaging_3(img2)
    img1_out = sharpening(img1)
    img2_out = sharpening(img2)
    cv2.imshow('img1',img1)
    cv2.imshow('img2',img2)
    cv2.imshow('img1_out',img1_out)
    cv2.imshow('img2_out',img2_out)
    # img1 = cv2.imread('./Q4_1.tif')
    # img2 = cv2.imread('./Q4_2.tif')
    #
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #
    # # img1
    # img1_laplacian_raw = laplacian(img1)
    # img1_laplacian = img1+img1_laplacian_raw
    # img1_laplacian = median_filter.median(7,img1_laplacian)
    # # img1_laplacian = median.reduce_SAP_12012724(3, img1_laplacian)
    #
    # img1_sobel_raw = sobel(img1)
    # img1_sobel = img1 + img1_sobel_raw
    # img1_sobel = averaging_filter.averaging_3(img1_sobel)
    # img1_combined = np.zeros(img1.shape,dtype=np.uint8)
    #
    # for i in range(img1.shape[0]):
    #     for j in range(img1.shape[1]):
    #
    #         img1_combined[i,j] = np.around(0.8*img1_laplacian[i,j] + 0.2*img1_sobel_raw[i,j])
    #         #img1_combined[i,j] = np.uint8(np.around(np.sqrt(np.uint16(img1_sobel[i,j])*np.uint16(img1_laplacian[i,j]))))
    # img1_combined = hist_match(img1_combined, img1)
    #
    #
    #
    # # img2
    # img2_average = averaging_filter.averaging_3(img2)
    # img2_laplacian_raw = laplacian(img2_average)
    # img2_laplacian = img2_average + img2_laplacian_raw
    # img2_laplacian = median_filter.median(7,img2_laplacian)
    # img2_sobel_raw = sobel(img2_average)
    # img2_sobel_raw = averaging_filter.averaging_3(img2_sobel_raw)
    # img2_sobel = img2_average + img2_sobel_raw
    # img2_combined = np.zeros(img2.shape,dtype = np.uint8)
    # for i in range(img2.shape[0]):
    #     for j in range(img2.shape[1]):
    #         img2_combined[i,j] = np.around(0.8*img2_laplacian[i,j] + 0.2*img2_sobel_raw[i,j])
    #
    # img2_combined = hist_match(img2_combined,img2_average)
    #
    #
    #
    # cv2.imshow('img1', img1)
    # cv2.imshow('img1_laplacian_raw', img1_laplacian_raw)
    # cv2.imshow('img1_laplacian',img1_laplacian)
    # cv2.imshow('img1_sobel_raw', img1_sobel_raw)
    # cv2.imshow('img1_sobel', img1_sobel)
    # cv2.imshow('img1_combined', img1_combined)
    #
    # cv2.imshow('img2', img2)
    # cv2.imshow('img2_average', img2_average)
    # cv2.imshow('img2_laplacian_raw', img2_laplacian_raw)
    # cv2.imshow('img2_laplacian',img2_laplacian)
    # cv2.imshow('img2_sobel_raw', img2_sobel_raw)
    # cv2.imshow('img2_sobel', img2_sobel)
    # cv2.imshow('img2_combined', img2_combined)
    cv2.waitKey(0)