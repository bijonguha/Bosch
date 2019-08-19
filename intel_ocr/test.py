# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:14:28 2019

@author: BIG1KOR
"""

from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
#%%
image = plt.imread('data/1.jpeg')
image.shape
plt.imshow(image)
#%%
gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')
#%%
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')
#%%
image = plt.imread('data/index.png')
plt.imshow(image)

#%%
# converting to grayscale
gray = rgb2gray(image)

# defining the sobel filters
sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
print(sobel_horizontal, 'is a kernel for detecting horizontal edges')
 
sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
print(sobel_vertical, 'is a kernel for detecting vertical edges')
#%%
out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
# here mode determines how the input array is extended when the filter overlaps a border.
#%%
plt.imshow(out_h, cmap='gray')
#%%
plt.imshow(out_v, cmap='gray')
#%%
pic = plt.imread('data/1.jpeg')/255  # dividing by 255 to bring the pixel values between 0 and 1
print(pic.shape)
plt.imshow(pic)
#%%
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
pic_n.shape
#%%

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]
#%%

cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)
#%%
#!/usr/bin/python3
# 2018.01.16 01:11:49 CST
# 2018.01.16 01:55:01 CST
import cv2
import numpy as np

## (1) read
img = cv2.imread("data/bijon.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## (2) threshold
th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

## (3) minAreaRect on the nozeros
pts = cv2.findNonZero(threshed)
ret = cv2.minAreaRect(pts)

(cx,cy), (w,h), ang = ret
if w>h:
    w,h = h,w
    ang += 90                                                                                                                                                                                                        

## (4) Find rotated matrix, do rotation
M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

## (5) find and draw the upper and lower boundary of each lines
hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)

th = 2
H,W = img.shape[:2]
uppers = np.array([y for y in range(H-1) if hist[y]<=th and hist[y+1]>th])
lowers = np.array([y for y in range(H-1) if hist[y]>th and hist[y+1]<=th])

diff = np.array([j-i for i,j in zip(uppers,lowers)])
diff_index = np.array([True if j > np.mean(diff) else False for j in diff ])

rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

uppers = uppers - int( np.mean(diff)/2 )
#%%
for y in uppers[diff_index]:
    cv2.line(rotated, (0,y), (W, y), (0,0,255), 3)

for y in lowers[diff_index]:
    cv2.line(rotated, (0,y), (W, y), (0,255,0), 3)

cv2.imwrite("result_line.png", rotated)
#%%
for left,right in zip(uppers[diff_index], lowers[diff_index]):
     cv2.rectangle(rotated ,(0,left),(W,right),(0,255,0),3)

cv2.imwrite("result_rect.png", rotated)
