import cv2
import numpy as np
import scipy as sci

def nearest_zly(img, dim):
    # numpy array expression of source image
    img1 = np.array(img)
    # numpy array expression initialization of target image
    img2 = np.zeros(dim, dtype=np.uint8)
    # spatial transformation
    x_scale = dim[0] / img.shape[0]
    y_scale = dim[1] / img.shape[1]
    for i in range(dim[0]):
        for j in range(dim[1]):
            # finding the nearest neighbor
            x = int(np.round(i / x_scale)-1)
            y = int(np.round(j / y_scale)-1)
            # assign the value of the nearest neighbor
            img2[i,j] = img1[x, y]
    return img2


def bilinear_zly(img,dim):
    # numpy array expression of source image
    img1 = np.array(img)

    img2 = np.zeros(dim,dtype=np.uint8)
    # spatial transformation
    x_scale = dim[0] / img.shape[0]# numpy array expression initialization of target image
    y_scale = dim[1] / img.shape[0]
    for i in range(dim[0]):
        for j in range(dim[1]):
            x = float(i) / x_scale
            y = float(j) / y_scale
            # finding the upper left neighbor
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))

            # boundary process -- to unilinear interpolation
            if x0 >= img.shape[0] - 1 or y0 >= img.shape[1] - 1 :
                if x0 >= img.shape[0] - 1 :
                    img2[i, j] = int((int(img1[img.shape[0]-1,y0+1]) - int(img1[img.shape[0]-1,y0])) * (y - y0) + int(img1[img.shape[0]-1, y0]))
                    break
                elif y0 >= img.shape[1] - 1:
                    img2[i, j] = int(int(img1[x0+1,img.shape[1]-1]) - int(img1[x0,img.shape[1]-1]) * (x - x0) + int(img1[x0, img.shape[1]-1]))
                    break
            # calculating the slope of horizontal dimension
            kx_1 = int(img1[x0+1,y0]) - int(img[x0,y0])
            kx_2 = int(img1[x0+1,y0+1]) - int(img[x0,y0+1])
            # calculating the value of R1, R2
            value_x1 = int(img1[x0,y0]) + kx_1*(x-x0)
            value_x2 = int(img1[x0,y0+1]) + kx_2*(x-x0)
            # calculating the slope of vertical dimentsion
            k_y = value_x2 - value_x1
            # calculating the value of P
            value = int(value_x1 + (y-y0)*k_y)
            # assigning the value
            img2[i, j] = value
    return img2


# kernel function definition
def kernel(x,a):
    y = np.zeros(4,float)
    for i in range(4):
            abs = np.abs(x[i])
            t = abs
            if abs <= 1 and abs > 0:
                y[i] = (a+2)*(t**3) - (a+3)*(t**2) + 1
            elif abs <=2 and abs > 1:
                y[i] = a*(t**3) - (5*a)*(t**2) + (8*a)*t - 4*a
            else:
                y[i] = 0
    return y


def bicubic_zly(img,dim):
    # setting the value of a in kernel function
    a = -0.75
    # numpy array expression of source image
    img1 = np.array(img)
    # numpy array expression initialization of target image
    img2 = np.zeros(dim,dtype=np.uint8)
    # a tiny disturbance added to the scales, to avoid mapping to zeros of kernel funciton.
    epsilon = 0.00001
    # scaling
    x_scale = dim[0] / img.shape[0] + epsilon
    y_scale = dim[1] / img.shape[1] + epsilon
    for i in range(dim[0]):
        for j in range(dim[1]):
            # spatial transformation
            x = float(i) / x_scale
            y = float(j) / y_scale
            # finding the nearest neighbor
            tmp_x = round(x)
            tmp_y = round(y)
            x0 = int(tmp_x)
            y0 = int(tmp_y)

            # the decimal part of x, y
            u = x - x0
            v = y - y0

            # determining the pattern of distribution of 16 nearest points.
            if u>=0:
                i_arr = np.array([-1, 0, 1, 2])
            elif u<0:
                i_arr = np.array([-2, -1, 0, 1])

            if v>=0:
                j_arr = np.array([-1, 0 ,1 ,2])
            elif v<0:
                j_arr = np.array([-2, -1, 0, 1])

            # vectorization of u,v
            u_arr = np.zeros(4,float)
            v_arr = np.zeros(4,float)
            for cnt in range(4):
                u_arr[cnt] = u
            for cnt in range(4):
               v_arr[cnt] = v
            # vectorization of x0,y0
            x0_arr = np.zeros(4,int)
            y0_arr = np.zeros(4,int)
            for cnt in range(4):
                x0_arr[cnt] = x0
            for cnt in range(4):
                y0_arr[cnt] = y0

            # boundary processing (a very preliminary one)
            flag = False

            if tmp_x >= img.shape[1] - 4 or tmp_y >= img.shape[0] -4:
                if tmp_x >= img.shape[1] - 4:
                    tmp_x = img.shape[1] -1
                    if tmp_y > img.shape[0] - 1:
                         tmp_y = img.shape[0] - 1
                    img2[i, j] = img1[img.shape[1]-1, tmp_y]
                    flag = True
                elif tmp_y >= img.shape[0] - 4:
                    tmp_y = img.shape[0]-1
                    if tmp_x > img.shape[1] - 1:
                         tmp_x = img.shape[0] -1
                    img2[i, j] = img1[tmp_x,img.shape[0] - 1]
                    flag = True

            if tmp_x <= 3 or tmp_y <= 3:
                if tmp_x <= 3:
                    img2[i, j] = img1[0, tmp_y]
                    flag = True
                elif tmp_y <= 3:
                    img2[i, j] = img1[tmp_x,0]
                    flag = True

            # non-boundary points calculation
            if flag == False :
                # construct the computational matrix
                temp = [
                       [img1[x0+i_arr[0],y0+j_arr[0]],img1[x0+i_arr[0],y0+j_arr[1]],img1[x0+i_arr[0],y0+j_arr[2]],img[x0+i_arr[0],y0+j_arr[3]]],
                       [img1[x0+i_arr[1],y0+j_arr[0]],img1[x0+i_arr[1],y0+j_arr[1]],img1[x0+i_arr[1],y0+j_arr[2]],img[x0+i_arr[1],y0+j_arr[3]]],
                       [img1[x0+i_arr[2],y0+j_arr[0]],img1[x0+i_arr[2],y0+j_arr[1]],img1[x0+i_arr[2],y0+j_arr[2]],img[x0+i_arr[2],y0+j_arr[3]]],
                       [img1[x0+i_arr[3],y0+j_arr[0]],img1[x0+i_arr[3],y0+j_arr[1]],img1[x0+i_arr[3],y0+j_arr[2]],img[x0+i_arr[3],y0+j_arr[3]]]
                        ]
                # calculation of matrix multiplication, and assignment of the value.
                img2[i, j] =np.dot(np.dot(kernel(u - i_arr , a) , temp) , kernel(v - j_arr , a).T)

    return img2


def nearest_cv(img,dim):
    return cv2.resize(img, dsize = (dim[1],dim[0]),fx = 1, fy = 1 , interpolation=cv2.INTER_NEAREST)
def bilinear_cv(img,dim):
    return cv2.resize(img, dsize = (dim[1],dim[0]),fx = 1, fy = 1 , interpolation=cv2.INTER_LINEAR)
def bicubic_cv(img,dim):
    return cv2.resize(img, dsize = (dim[1],dim[0]),fx =1, fy = 1 , interpolation=cv2.INTER_CUBIC)



img = cv2.imread('./rice.tif', cv2.IMREAD_GRAYSCALE)
source_size = img.shape;
target_size = [source_size[0]*2,source_size[1]*2];
print(source_size);
print(target_size);
img2 = nearest_zly(img,[int(target_size[0]),int(target_size[1])])
img3 = bilinear_zly(img,[int(target_size[0]),int(target_size[1])])
img4 = bicubic_zly(img,[int(target_size[0]),int(target_size[1])])
img5 = nearest_cv(img,[int(target_size[0]),int(target_size[1])])
img6 = bilinear_cv(img,[int(target_size[0]),int(target_size[1])])
img7 = bicubic_cv(img,[int(target_size[0]),int(target_size[1])])
cv2.imshow('img1', img)
cv2.imshow('nearest', img2)
cv2.imshow('bilinear', img3)
cv2.imshow('bicubic',img4)
cv2.imshow('cv_nearest',img5)
cv2.imshow('cv_bilinear',img6)
cv2.imshow('cv_bicubic',img7)
cv2.waitKey(0)


