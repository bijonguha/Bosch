# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:07:56 2019

@author: DMV4KOR
"""
import cv2

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
#%%
def extract_box(img):
    """
    Function to extract the boxes in the ruled worksheet
    Output:
        c = contour ids
        contours = contours
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Otsu thresholding
    t, binary_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    
    # Defining a kernel length
    kernel_length = np.array(binary_image).shape[1]//80
     
    
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(binary_image, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(binary_image, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=4)
    
    #Join horizontal and vertical images
    alpha = 0.5
    beta = 1.0 - alpha
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    #Find and sort the contours
    im2, contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    
    c = []
    s = []
    area = []
    for contour in contours:
        area.append(cv2.contourArea(contour))
    s = np.argsort(area)
    
    #Find the correct boxes where area is between 40% and 45% of the largest rectangle
    for i,contour in enumerate(contours):
        if cv2.contourArea(contour) >= area[s[-1]]*0.40  and cv2.contourArea(contour) < area[s[-1]]*0.45:
            c.append(i)
    
    return c, contours
#%%
img = cv2.imread("data/box+lines2.jpg")
#img = cv2.imread("ruled+box.JPG")
c,contours =  extract_box(img)
#%%
# Draw contours

cnt = contours[0]
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
bb = cv2.drawContours(img,contours,-1,(0,0,255),2)
plt.figure(figsize=(20,20))
plt.imshow(bb, cmap="gray")

#%%
# writing the output
cropped_dir_path = "C:/Users/DMV4KOR/Desktop/"
idx = 0
for i in c:
    # Returns the location and width,height for every contour
    x, y, w, h = cv2.boundingRect(contours[i])
    
    idx += 1
    new_img = img[y:y+h, x:x+w]
    cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
#%%
"""
To do:
    * Find a generic area detection
    * Test with some more images
"""