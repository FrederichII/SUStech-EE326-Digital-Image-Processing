import numpy as np
import cv2

def padding(img,kernel_size):
    # 获取图片的高，宽
    img_height, img_width = img.shape
    # 获得 K 值
    kx, ky = kernel_size[1] // 2, kernel_size[0] // 2

    # 左右需要填充的 zero
    # 此处必须加上 dtype=np.uint8，否则 cv2 float 显示全白
    zeros_array = np.zeros((img_height, kx), dtype=np.uint8)
    # 给图片左右两边添加 0
    img_copy = np.concatenate([zeros_array, img, zeros_array], axis=1)
    # 上下需要填充的 zero
    zeros_array = np.zeros((ky, 2 * kx + img_width), dtype=np.uint8)
    # 给图片上下两边添加 0
    img_result = np.concatenate([zeros_array, img_copy, zeros_array], axis=0)
    return img_result


def padding_upleft(img):
    output = np.zeros([img.shape[0]*2,img.shape[1]*2], dtype=np.uint8)
    output[0:img.shape[0], 0:img.shape[1]] = img
    return output