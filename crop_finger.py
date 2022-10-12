import cv2
import math
import numpy as np
import streamlit as st
from PIL import Image
import torch, torchvision

#image manipulations 

class ResizeMe(object):
    #resize and center image in desired size 
    def __init__(self,desired_size):
        
        self.desired_size = desired_size
        
    def __call__(self,img):
    
        img = np.array(img).astype(np.uint8)
        
        desired_ratio = self.desired_size[1] / self.desired_size[0]
        actual_ratio = img.shape[0] / img.shape[1]

        desired_ratio1 = self.desired_size[0] / self.desired_size[1]
        actual_ratio1 = img.shape[1] / img.shape[0]

        if desired_ratio < actual_ratio:
            img = cv2.resize(img,(int(self.desired_size[1]*actual_ratio1),self.desired_size[1]),None,interpolation=cv2.INTER_AREA)
        elif desired_ratio > actual_ratio:
            img = cv2.resize(img,(self.desired_size[0],int(self.desired_size[0]*actual_ratio)),None,interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img,(self.desired_size[0], self.desired_size[1]),None, interpolation=cv2.INTER_AREA)
            
        h, w, _ = img.shape

        new_img = np.ones((self.desired_size[1],self.desired_size[0],3))
        
        hh, ww, _ = new_img.shape

        yoff = int((hh-h)/2)
        xoff = int((ww-w)/2)
        
        new_img[yoff:yoff+h, xoff:xoff+w,:] = img

        
        return Image.fromarray(new_img.astype(np.uint8))

class MakeLandscape():
    #flip if needed
    def __init__(self):
        pass
    def __call__(self,img):
        
        if img.height> img.width:
            img = np.rot90(np.array(img), k=3)
            #img = np.array(img)
            img = Image.fromarray(img)
            #img = cv2.imread(img)
        return img

#function to crop the image to boxand rotate

def get_cropped(rotrect,box,image):
    
    width = int(rotrect[1][0])
    height = int(rotrect[1][1])

    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def calculateDistance(x1,y1,x2,y2):  
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist

@st.cache
def get_cropped_finger(img,predictor,return_mapping=False,resize=None):
    #convert to numpy    
    img = np.array(img)[:,:,::-1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    #get prediction
    outputs = predictor(img)
    
    #get boxes and masks
    ins = outputs["instances"]
    pred_masks = ins.get_fields()["pred_masks"]
    boxes = ins.get_fields()["pred_boxes"]    
    
    #get main leaf mask if the area is >= the mean area of boxes and is closes to the centre 
    
    masker = pred_masks[np.argmin([calculateDistance(x[0], x[1], int(img.shape[1]/2), int(img.shape[0]/2)) for i,x in enumerate(boxes.get_centers()) if (boxes[i].area()>=torch.mean(boxes.area()).to("cpu")).item()])].to("cpu").numpy().astype(np.uint8)

    #mask image
    mask_out = cv2.bitwise_and(img, img, mask=masker)
    
    #find contours and boxes
    contours, hierarchy = cv2.findContours(masker.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[np.argmax([cv2.contourArea(x) for x in contours])]
    rotrect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)
    

    #crop image
    cropped = get_cropped(rotrect,box,mask_out)

    #resize
    rotated = MakeLandscape()(Image.fromarray(cropped))
    
    if not resize == None:
        resized = ResizeMe((resize[0],resize[1]))(rotated)
    else:
        resized = rotated
        
    if return_mapping:
        img = cv2.drawContours(img, [box], 0, (0,0,255), 10)
        img = cv2.drawContours(img, contours, -1, (255,150,), 10)
        return resized, ResizeMe((int(resize[0]),int(resize[1])))(Image.fromarray(img))
    
    return resized

