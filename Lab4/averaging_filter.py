import numpy as np
import cv2

def averaging_3(img):
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

    output_img = np.zeros(enlarged_img.shape,dtype=np.uint8)
    for i in range(1, enlarged_img.shape[0] - 1):
        for j in range(1, enlarged_img.shape[1] - 1):
            tmp = enlarged_img
            frame = np.array([[tmp[i - 1, j - 1], tmp[i - 1, j], tmp[i - 1, j + 1]], [tmp[i, j - 1], tmp[i, j], tmp[i, j + 1]],[tmp[i + 1, j - 1], tmp[i + 1, j], tmp[i + 1, j + 1]]])
            output_img[i,j] = np.sum(frame)/9
    result = output_img[1:output_img.shape[0]-1,1:output_img.shape[1]-1]
    return result
