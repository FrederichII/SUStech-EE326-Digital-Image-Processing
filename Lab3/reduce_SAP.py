import numpy as np
import cv2

def reduce_SAP_12012724(window_size, img):
    remain = int(window_size / 2)
    height = img.shape[0]
    width = img.shape[1]

    # padding
    # --4 edges
    # --4 corners

    #--edges
    enlarged_img = np.zeros([height+2*remain, width+2*remain])
    enlarged_img[remain:height+remain,remain:width+remain] = img
    enlarged_img[0:remain, remain:width + remain] = img[0:remain,:]
    enlarged_img[enlarged_img.shape[0]-remain:enlarged_img.shape[0], remain:width + remain] = img[img.shape[0]-remain:img.shape[0],:]
    enlarged_img[remain:height+remain, 0:remain] = img[:, 0:remain]
    enlarged_img[remain:height+remain, enlarged_img.shape[1]-remain:enlarged_img.shape[1]] = img[:,img.shape[1]-remain:img.shape[1]]

    #--corners
    enlarged_img[0:remain,0:remain] = img[0:remain,0:remain]
    enlarged_img[0:remain,enlarged_img.shape[1]-remain:enlarged_img.shape[1]] = img[0:remain,img.shape[1]-remain:img.shape[1]]
    enlarged_img[enlarged_img.shape[0]-remain:enlarged_img.shape[0],0:remain] = img[img.shape[0]-remain:img.shape[0],0:remain]
    enlarged_img[enlarged_img.shape[0]-remain:enlarged_img.shape[0],enlarged_img.shape[1]-remain:enlarged_img.shape[1]] = img[img.shape[0]-remain:img.shape[0],img.shape[1]-remain:img.shape[1]]


    temp_img = np.zeros(enlarged_img.shape,dtype=np.uint8)
    for i in range(remain,height+remain):
        for j in range(remain,width+remain):
            median = np.median(enlarged_img[i-remain:i+remain+1,j-remain:j+remain+1])
            temp_img[i, j] = median

    img = temp_img[remain:height+remain, remain:width+remain]

    return img

if __name__ == '__main__':
    img = cv2.imread('./Q3_4.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img_3 = reduce_SAP_12012724(3, img)
    filtered_img_5 = reduce_SAP_12012724(5, img)
    filtered_img_7 = reduce_SAP_12012724(7, img)
    cv2.imshow('img',img)
    cv2.imshow('filtered_img_3', filtered_img_3)
    cv2.imshow('filtered_img_5', filtered_img_5)
    cv2.imshow('filtered_img_7', filtered_img_7)
    cv2.waitKey(0)
