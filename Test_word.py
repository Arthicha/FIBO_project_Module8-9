__author__ = 'Zumo'
# this is only test program

import cv2
import copy
import numpy as np
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP



def Get_Plate(img,sauvola_kernel=11,perc_areaTh=[0.001,0.5] ,minimumLength=0.1,plate_opening=7,char_closing=13,Siz=80.0):

    org = copy.deepcopy(img)
    x,y,c = org.shape
    areaTh=(perc_areaTh[0]*x*y,perc_areaTh[1]*x*y)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, img = cv2.threshold(gray, 127, 255,  cv2.THRESH_OTSU)
    img = IP.binarize(gray,IP.SAUVOLA_THRESHOLDING,sauvola_kernel)
    img = np.array(img,dtype=np.uint8)
    org = copy.deepcopy(img)

    img_c = copy.deepcopy(img)
    img_c = IP.morph(img_c, mode=IP.OPENING, value=[plate_opening, plate_opening])


    #cv2.imshow('frame',img_c)
    #cv2.waitKey(0)
    img_c, contours, hierarchy = cv2.findContours(img_c, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    subImg = []


    for cnt in contours:
        epsilon = minimumLength*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        area = cv2.contourArea(cnt)
        if (len(approx) == 4) and (area > areaTh[0]) and (area < areaTh[1]):
            img_p = IP.remove_perspective(img,approx,(int(Siz),int(Siz)),org_shape=(x,y))
            white = np.count_nonzero(img_p)/(Siz*Siz)
            if (white > 0.1):
                img_m = IP.morph(img_p,mode=IP.CLOSING,value=[char_closing,char_closing])
                pern = 100
                for i in range(0,10):
                    img_f = IP.magnifly(img_m,percentage=pern)
                    a = np.amin(np.count_nonzero(img_f,axis=0)/(Siz))
                    b = np.amin(np.count_nonzero(img_f,axis=1)/(Siz))
                    if b > a:
                        a = b
                    if a < 0.04:
                        break
                    pern += 10
                img_p = IP.magnifly(img_p,percentage=pern)
                for angle in [0,90,180,-90]:
                    img_r = IP.rotation(img_p,(img_p.shape[0]/2,img_p.shape[1]/2),angle)
                subImg.append(img_p)
                cv2.drawContours(org,[approx],0,(0,0,255),3)

    return org,subImg

cap = cv2.VideoCapture(1)
while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    org = copy.deepcopy(frame)
    # Our operations on the frame come here

    org,LoM = Get_Plate(frame)
    # Display the resulting frame
    for i in range(0,len(LoM)):
        cv2.imshow('output'+str(i),LoM[i],)
        cv2.moveWindow('output'+str(i),100*i,80)
    cv2.imshow('original',org)
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break
    for i in range(0,len(LoM)):
        cv2.destroyWindow('output'+str(i))
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
