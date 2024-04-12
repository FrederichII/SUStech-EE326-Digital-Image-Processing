import cv2
import numpy as np
import matplotlib.pyplot as plt
import histogram_equalization as hseq


def local_hist_12012724(img, seg_size):
    shape = img.shape
    row_num = int(shape[0] / seg_size[0])
    col_num = int(shape[1] / seg_size[1])
    row_remain = shape[0] % seg_size[0]
    col_remain = shape[1] % seg_size[1]
    rows = []
    cols = []
    seg_height = seg_size[0]
    seg_width = seg_size[1]
    for i in range(row_num):
        rows.append([i*seg_height, (i+1)*seg_height])
    for i in range(col_num):
        cols.append([i*seg_width,(i+1)*seg_width])
    if row_remain != 0:
        row_num += 1
        rows.append([shape[0] - row_remain,shape[0]])
    if col_remain != 0:
        col_num += 1
        cols.append([shape[1] - col_remain, shape[1]])

    output_img = np.zeros(shape, np.uint8)
    for i in range(row_num):
        for j in range(col_num):
            output_img[rows[i][0]:rows[i][1],cols[j][0]:cols[j][1]], output_hist, input_hist= hseq.hist_equ_12012724(img[rows[i][0]:rows[i][1],cols[j][0]:cols[j][1]])


    return output_img


if __name__ == '__main__':
    img = cv2.imread('./Q3_3.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global_img,output_hist,input_hist= hseq.hist_equ_12012724(img)
    local_img = local_hist_12012724(img, [5,5])
    cv2.imshow('input image',img)
    cv2.imshow('global equalization', global_img)
    cv2.imshow('local equalization', local_img)
    cv2.waitKey()