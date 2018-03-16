__author__ = 'Zumo'
# this is only test program

import cv2
import copy
import numpy as np
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP

def checkOreantation(img):

    LMG = []
    for name in ['5','twoTH','ThreeEN']:
        sample = cv2.imread(name+'.jpg')
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        ret, sample = cv2.threshold(sample, 127, 255,0)
        sample, contours, hierarchy = cv2.findContours(sample, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        sample = []
        for cnt in contours:
            sample += cnt.tolist()
        sample = np.array(sample)
        LMG.append(sample)
    img, contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    img = []
    for cnt in contours:
        img += cnt.tolist()
    img = np.array(img)

    p_ret = cv2.matchShapes(img,LMG[0],1,0.0)
    for i in range(1,len(LMG)):
        ret = cv2.matchShapes(img,LMG[i],1,0.0)
        if ret < p_ret:
            p_ret = ret
    return ret


def aspectRatio(img_f):
    img_fc = copy.deepcopy(img_f)
    img_fc, cfc, hfc = cv2.findContours(img_fc, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    try:
        xfc,yfc,wfc,hfc = cv2.boundingRect(cfc[-1])
    except:
        return 1.0
    aspect_ratio = float(wfc)/hfc
    return aspect_ratio

def getWordSize(img_f):
    img_fc = copy.deepcopy(img_f)
    img_fc, contours, hfc = cv2.findContours(img_fc, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    try:
        cnt = contours[-1]
    except:
        return 0,0
    leftmost = np.array(cnt[cnt[:,:,0].argmin()][0])
    rightmost = np.array(cnt[cnt[:,:,0].argmax()][0])
    topmost = np.array(cnt[cnt[:,:,1].argmin()][0])
    bottommost = np.array(cnt[cnt[:,:,1].argmax()][0])
    return np.linalg.norm(leftmost-rightmost),np.linalg.norm(topmost-bottommost)


def Get_Plate(img,sauvola_kernel=11,perc_areaTh=[0.005,0.5] ,numberOword=(0.5,1.5),minimumLength=0.05,plate_opening=3,char_opening=13,Siz=60.0):

    org = copy.deepcopy(img)
    x,y,c = org.shape
    areaTh=(perc_areaTh[0]*x*y,perc_areaTh[1]*x*y)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_o = cv2.threshold(gray, 127, 255,  cv2.THRESH_OTSU)
    img_s = IP.binarize(gray,IP.SAUVOLA_THRESHOLDING,sauvola_kernel)
    img_s = np.array(img_s,dtype=np.uint8)
    img = cv2.bitwise_and(img_s,img_o)


    img_c = copy.deepcopy(img)
    img_c = IP.morph(img_c, mode=IP.ERODE, value=[plate_opening, plate_opening])
    #org = copy.deepcopy(img_c)

    #cv2.imshow('frame',img_c)
    #cv2.waitKey(0)
    img_c, contours, hierarchy = cv2.findContours(img_c, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    subImg = []

    for ic in range(0,len(contours)):
        cnt = contours[ic]
        hi = hierarchy[0][ic]
        epsilon = minimumLength*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        area = cv2.contourArea(cnt)
        if (len(approx) == 4) and (hi[0] != -1) and (hi[1] != -1) and (area > areaTh[0]) and (area < areaTh[1]):

            img_p = IP.remove_perspective(img,approx,(int(Siz),int(Siz)),org_shape=(x,y))
            white = np.count_nonzero(img_p)/(Siz*Siz)
            if (white > 0.1):
                img_m = IP.morph(img_p,mode=IP.OPENING,value=[char_opening,char_opening])

                aspect_ratio = aspectRatio(img_m)
                sz = min(getWordSize(img_m))
                if (aspect_ratio > numberOword[0]) and (aspect_ratio < numberOword[1]):
                    if aspect_ratio < 1.00:
                        rotating_angle = [0,180]
                    else:
                        rotating_angle = [90,-90]
                else:
                    if aspect_ratio < 1.00:
                        rotating_angle = [90,-90]
                    else:
                        rotating_angle = [0,180]
                diff = [0,0]


                for a in range(0,len(rotating_angle)):
                    angle = rotating_angle[a]
                    img_r = IP.rotation(img_p,(img_p.shape[0]/2,img_p.shape[1]/2),angle)
                    ctr = int(Siz/2)
                    img_r = img_r[ctr-15:ctr+15,ctr-30:ctr+30]
                    subImg.append(img_r)
                    '''chkO = checkOreantation(img_r)
                    diff[a] = [chkO,copy.deepcopy(img_r)]
                if diff[0][0] > diff[1][0]:
                    subImg.append(diff[0][1])
                else:
                    subImg.append(diff[1][1])'''
                cv2.drawContours(org,[approx],0,(0,0,255),3)

    return org,subImg


if 1:
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
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        for i in range(0,len(LoM)):
            cv2.destroyWindow('output'+str(i))
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
else:
    checkOreantation([])
    cap = cv2.VideoCapture(1)
    p_cnt = None
    chk = False
    while(True):
        ret, frame = cap.read()
        org = copy.deepcopy(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, img_o = cv2.threshold(gray, 127, 255,  cv2.THRESH_OTSU)
        img_s = IP.binarize(gray,IP.SAUVOLA_THRESHOLDING,13)
        img_s = np.array(img_s,dtype=np.uint8)
        img = cv2.bitwise_and(img_s,img_o)
        img_c, contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        a = []
        for cnt in contours:
            a += cnt.tolist()
        a = np.array(a)
        if chk:
            ret = cv2.matchShapes(a,p_cnt,1,0.0)
            print(ret)
        p_cnt = a
        chk = True
