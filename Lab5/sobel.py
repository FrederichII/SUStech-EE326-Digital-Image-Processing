import numpy as np
import cv2
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