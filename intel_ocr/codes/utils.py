#Loading libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
#%%
def extract_line(img, ruled = False):
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
#%%
#img = cv2.imread("data/bijon.jpg")
#extract_line(img)
#%%
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

def text_segment(img):
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
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    char_locs = []
    # For each contour, find the bounding rectangle and draw it
    x1l = y1l = x2l = y2l = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if ( y > y1l and (y+w) < y2l and x > x1l and (x+w) < x2l):
            continue
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)
            x1l = x
            y1l = y
            x2l = x + w
            y2l = y + w
            char_locs.append([x1l,y1l,x2l,y2l])

    plt.figure()    
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    return char_locs
#%%
#def rotateImage(image, angle):
#    row,col = image.shape[:2]
#    center=tuple(np.array([row,col])/2)
#    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
#    new_image = cv2.warpAffine(image, rot_mat, (col,row))
#    return new_image

image = cv2.imread("data/bijon.jpg")
image = cv2.copyMakeBorder(image,10,10,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

H,W = image.shape[:2]

y1s,y2s = extract_line(image)
x1s = 0
x2s = W

for i,j in zip(y1s,y2s):
    print(i,j)
    line_img = image[i:j, x1s:x2s]
    char_locs = text_segment(line_img)
    Nchar_locs = [[pt[0]+0,pt[1]+i,pt[2]+0,pt[3]+j] for pt in char_locs]