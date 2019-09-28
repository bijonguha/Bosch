# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#import imutils


# In[ ]:


import tensorflow as tf


# In[87]:


def extract_line(img, ruled = False):
    '''
    extract_line : Function to extracts the line from the image    
    argument:
        img (array): image array
        ruled(bool) : whether the input image is ruled or not
    output:
        uppers[diff_index]  : Upper points (x,y)
        lowers[diff_index]  : lower points(x,y)
    '''
    #Converting image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Binary thresholding and inverting at 127
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    
    #minAreaRect on the nozeros
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)
    
    (cx,cy), (w,h), ang = ret
    if w>h:
        w,h = h,w
        ang += 90     

    ##If page is rotated Find rotated matrix, do rotation
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
    
    uppers[1:] = [i-int(j)/3 for i,j in zip(uppers[1:], diff[1:])]
    lowers[:-1] = [i+int(j)/3 for i,j in zip(lowers[:-1], diff[:-1])]
    
#    uppers[1:] = uppers[1:] - int( np.mean(diff)/3 )
#    lowers[:-1] = lowers[:-1] + int( np.mean(diff)/3 )    
    
    for left,right in zip(uppers[diff_index], lowers[diff_index]):
        print(left,right)
        cv2.rectangle(rotated ,(0,left),(W,right),(0,255,0),5)
    
    plt.axis("off")
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.show()
    
    return uppers[diff_index], lowers[diff_index]


# In[88]:


img = cv2.imread("data/bijon.jpg")
extract_line(img)


# In[89]:


def sort_contours(cnts, method="left-to-right"):
    '''
    sort_contours : Function to sort contours
    argument:
        cnts (array): image contours
        method(string) : sorting direction
    output:
        cnts(list): sorted contours
        boundingBoxes(list): bounding boxes
    '''
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


# In[225]:


def text_segment(img,W,H):
    '''
    text_segment : Function to segment the characters
    argument:
        img (array): image array
        W(int)     : Width
        H(int)     : height
    output:
        char_locs(list)  : character locations
        char_type(list)  : character type(exponential or not)
    '''
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # smooth the image to avoid noises
    gray = cv2.medianBlur(gray,5)
    
    # Apply adaptive threshold
    ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    
    ## apply some dilation and erosion to join the gaps
    thresh = cv2.dilate(thresh,None,iterations = 3)
    #thresh = cv2.erode(thresh,None,iterations = 2)
    
    # Find the contours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted,bounding_boxes = sort_contours(contours,method="left-to-right")
    char_locs = []
    
    # For each contour, find the bounding rectangle and draw it
    contours_sorted = list(contours_sorted)
    x1l = y1l = x2l = y2l = 0

    i = 0
    char_type =[]
    while i in range(0, len(contours_sorted)):
            x,y,w,h = bounding_boxes[i]
            exp = 0
#            print(x,y,w,h," ",i)
            if i+1 != len(contours_sorted):
                x1,y1,w1,h1 = bounding_boxes[i+1]
                if abs(x-x1) < 20:
                    
                    minX = min(x,x1)
                    minY = min(y,y1)
                    maxX = max(x+w, x1+w1)
                    maxY = max(y+h, y1+h1)
                    x,y,x11,y11 = minX, minY, maxX, maxY
                    
                    x,y,w,h = x,y,x11-x,y11-y
                    i = i+1
                    

            if ( y > y1l and (y+w) < y2l and x > x1l and (x+w) < x2l):
                continue
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)
                x1l = x
                y1l = y
                x2l = x + w
                y2l = y + h
                char_locs.append([x,y,x+w,y+h])
#                image = draw_contour(img,contours_sorted[i] , i)
            if y+h < (H*(1/2)):
                exp = 1
            i = i+1
            char_type.append(exp)
    plt.figure(figsize=(12,12))    
    plt.axis("on")
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    return char_locs,char_type


#%%
from tensorflow.keras.preprocessing import image
model = tf.keras.models.load_model('my_model.h5')
#%%
def predict(img,x1,y1,x2,y2):
    '''
    predict  : Function to predict the character
    argument:
        x1,y1(int,int)    : Top left corner point
        x2,y2(int,int)    : Bottom right corner point
    output:
        c[index](int) : predicted character 

    '''
    new_img = img[y1:y2, x1:x2]
    new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    t, binary_image = cv2.threshold(new_img, 100, 255, cv2.THRESH_BINARY_INV)
    constant = cv2.copyMakeBorder(binary_image,40,40,40,40,cv2.BORDER_CONSTANT,value=0)
    cv2.imwrite("C:/Users/DMV4KOR/Desktop/4_2.jpg",constant)
    ing=tf.keras.preprocessing.image.load_img("C:/Users/DMV4KOR/Desktop/4_2.jpg", target_size=(28, 28))
    x=image.img_to_array(ing)
    x=np.expand_dims(ing, axis=0)
    images = np.vstack([x])
    
    classes = model.predict(images, batch_size=2)
    index = np.argmax(classes[0])
#    
    c = ['0','1','2','3','4','5','6','7','8','9','+','-','times','(',')']
    return c[index]

# In[226]:


image1 = cv2.imread("data/bijon.jpg")
image1 = cv2.copyMakeBorder(image1,10,10,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.show()

H,W = image1.shape[:2]

y1s,y2s = extract_line(image1)
x1s = 0
x2s = W

for i,j in zip(y1s,y2s):
   
    line_img = image1[i:j, x1s:x2s]
    lm = line_img.copy()
    h,w = line_img.shape[:2]
    
    char_locs,char_type = text_segment(lm,w,h)
    print(char_type)
    char_predict = []
    f = 0
  
    for x1,y1,x2,y2 in char_locs: 
        char_predict.append(predict(line_img,x1,y1,x2,y2))
    Nchar_locs = [[pt[0]+0,pt[1]+i,pt[2]+0,pt[3]+j] for pt in char_locs]
    


