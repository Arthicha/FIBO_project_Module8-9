__author__ = ['Zumo', 'Tew', 'Wisa']
__version__ = 1.11
# Fin all need function no comment though will add later
#                                           BY TEW

import scipy
import sys
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageFont, ImageDraw
from Foundation import Binarization, Filter


class Image_Processing_And_Do_something_to_make_Dataset_be_Ready():
    # def __init__(self):
    # GLOBAL
    SHAPE = (640, 480)
    CREATE_SHAPE = (320, 320)

    # Binarization Mode
    OTSU_THRESHOLDING = 0
    ADAPTIVE_CONTRAST_THRESHOLDING = 1
    NIBLACK_THRESHOLDING = 2
    SAUVOLA_THRESHOLDING = 3
    BERNSEN_THRESHOLDING = 4
    LMM_THRESHOLDING = 5

    # Noise Filter
    KUWAHARA = 6
    WIENER = 7
    MEDIAN = 8
    # Blur Filter
    GAUSSIAN = 9
    AVERAGING = 10
    BILATERAL = 11

    # morphology method
    ERODE = cv2.MORPH_ERODE
    DILATE = cv2.MORPH_DILATE
    OPENING = cv2.MORPH_OPEN
    CLOSING = cv2.MORPH_CLOSE
    GRADIENT = cv2.MORPH_GRADIENT
    TOP_HAT = cv2.MORPH_TOPHAT
    BLACK_HAT = cv2.MORPH_BLACKHAT

    # Estimate method
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_AREA = cv2.INTER_AREA
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4
    INTER_MAX = cv2.INTER_MAX
    WARP_FILL_OUTLIERS = cv2.WARP_FILL_OUTLIERS
    WARP_INVERSE_MAP = cv2.WARP_INVERSE_MAP

    # Generate Border Method
    BORDER_CONSTANT = cv2.BORDER_CONSTANT
    BORDER_REPLICATE = cv2.BORDER_REPLICATE
    BORDER_REFLECT = cv2.BORDER_REFLECT
    BORDER_WRAP = cv2.BORDER_WRAP
    BORDER_REFLECT101 = cv2.BORDER_REFLECT101
    BORDER_TRANSPARENT = cv2.BORDER_TRANSPARENT
    BORDER_ISOLATED = cv2.BORDER_ISOLATED

    # Colorcode
    GRAY_SCALE = cv2.COLOR_BGR2GRAY
    RGB = cv2.COLOR_BGR2RGB
    HSV = cv2.COLOR_BGR2HSV
    BGR = None

    # pass


    def binarize(image, method=OTSU_THRESHOLDING, value=None):
        if method == __class__.OTSU_THRESHOLDING:
            img = Binarization.Binarization_Otsu(image)
        elif method == __class__.ADAPTIVE_CONTRAST_THRESHOLDING:
            img = Binarization.Binarization_Adapt(image, value[0], value[1])
        elif method == __class__.NIBLACK_THRESHOLDING:
            img = Binarization.Binarization_Niblack(image, value[0], value[1])
        elif method == __class__.SAUVOLA_THRESHOLDING:
            img = Binarization.Binarization_Sauvola(image, value)
        elif method == __class__.BERNSEN_THRESHOLDING:
            img = Binarization.Binarization_Bernsen(image)
        elif method == __class__.LMM_THRESHOLDING:
            img = Binarization.Binarization_LMM(image)
        else:
            sys.exit("Unknown method\n")
        return img.astype(np.uint8)
    # Binarize image into two value 255 or 0
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.binarize(img,method=ipaddr.OTSU_THRESHOLDING)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def remove_noise(image, method=KUWAHARA, value=5):
        if method == __class__.KUWAHARA:
            img = Filter.Filter_Kuwahara(image, value)
        elif method == __class__.WIENER:
            img = Filter.Filter_weiner(image, window_size=value[0], iteration=value[1])
        elif method == __class__.MEDIAN:
            img = Filter.Filter_median(image, window=value)
        else:
            sys.exit("Unknown method\n")
        return img
    # reduce image noise such as salt and pepper noise
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.remove_noise(img,method=ipaddr.KUWAHARA , value=5)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def resize(image, shape=SHAPE, method=INTER_LINEAR):
        img = cv2.resize(image, shape, interpolation=method)
        return img
    # change image size
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.resize(img,(28,28) )
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def capture(cam, size=SHAPE, mode=GRAY_SCALE, config=None):
        ret, image = cam.read()
        if mode != __class__.BGR:
            image = cv2.cvtColor(image, mode, config)
        image = cv2.resize(image, size)
        return image
    # change image size
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       camera = cv2.videoCapture(0)
       img = ipaddr.capture( camera)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def translate(image, value, config=None):
        matrix = np.float32([[1, 0, value[0]], [0, 1, value[1]]])
        if config is None:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=image.shape)
        elif config[1] == __class__.BORDER_CONSTANT:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=image.shape, flags=config[0],
                                 borderMode=config[1], borderValue=config[2])
        else:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=image.shape, flags=config[0],
                                 borderMode=config[1])
        return img
    # translate image (move to the left right or whatever by value)
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.translate(img,(1,1))
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def blur(image, method=AVERAGING, value=5):
        if method == __class__.MEDIAN:
            img = cv2.medianBlur(image, value)
        elif method == __class__.AVERAGING:
            averaging_kernel = np.ones([value, value], dtype=np.float32) / pow(value, 2)
            img = cv2.filter2D(image, -1, averaging_kernel)
        elif method == __class__.GAUSSIAN:
            img = cv2.GaussianBlur(image, (value, value), 0)
        elif method == __class__.BILATERAL:
            img = cv2.bilateralFilter(image, value[0], value[1], value[2])
        else:
            sys.exit("Unknown method\n")
        return img
    # blur image with three method
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.blur(img,ipaddr.AVERAGING,3)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def morph(image, mode=DILATE, value=[3, 3]):
        matrix = np.ones((value[0], value[1]), np.float32)
        img = cv2.morphologyEx(image, mode, matrix)
        return img
    # morph image acording to mode and value use to construct kernel
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.morph(img,ipaddr.DILATE,[15,15])
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def remove_perspective(image, region, shape,org_shape=None):
        if org_shape == None:
            org_shape = shape
        #print([region[3], region[1], region[2], region[0]])
        #pts1 = np.float32([region[2], region[3], region[1], region[0]])
        pts2 = np.float32([[0, 0], [org_shape[0], 0], [org_shape[0], org_shape[1]], [0, org_shape[1]]])

        best_pts = []
        min_cost = pow(shape[0]*shape[1],2)
        for i in range(0,4):
            rg = np.reshape(region,(-1,2)).tolist()
            pts_1 = np.array(rg[-i:] + rg[:-i])
            pts_2 = np.array(pts2)
            cost = np.sum(np.abs(pts_1-pts_2))
            if min_cost >= cost:
                min_cost = cost
                best_pts = pts_1
        pts_1 = best_pts
        pts2 = np.float32([[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]])

        pts1 = np.float32([[pts_1[0]],[pts_1[1]],[pts_1[2]],[pts_1[3]]])
        #print(pts1.tolist())
        #pts1 = np.float32([region[1], region[0], region[3], region[2]])
        #print([[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(image, matrix, shape,borderValue=255)
        return img
    # morph image acording to mode and value use to construct kernel
    # region value is box containing data that u want to remove perspective like [()]
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.morph(img,ipaddr.DILATE,[15,15])
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def rotation(image, center_of_rotation, angle):
        matrix = cv2.getRotationMatrix2D((center_of_rotation[0], center_of_rotation[1]), angle, 1)
        # print(matrix)
        # print(image.shape)
        img = cv2.warpAffine(image, matrix, (image.shape[1],image.shape[0]), borderMode=__class__.BORDER_CONSTANT,
                             borderValue=255)
        # print(img.shape)
        return img
    # rotate image according to center of rotation and angle
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.rotation(img,(img.shape[1]/2,img.shape[2]/2),15)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def font_to_image(font, size=CREATE_SHAPE, index=0, string="0"):
        # Create Plate from font and word
        # Example
        # from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
        # image = ipaddr.font_to_image("angsana.ttc", 10, 0, "หนึ่ง")
        # cv2.imshow("one", image)
        # cv2.waitKey(0)

        Text_Font = ImageFont.truetype(font, size, index, encoding="unic")
        w, h = Text_Font.getsize(string)
        img = Image.new("L", __class__.CREATE_SHAPE, color=255)
        image = ImageDraw.Draw(img)
        image.text(((__class__.CREATE_SHAPE[0] - w) / 2, (__class__.CREATE_SHAPE[1] - h) / 2), string, font=Text_Font,
                   fill="black")
        img = np.array(img)
        cv2.rectangle(img, (60, 60), (__class__.CREATE_SHAPE[0] - 60, __class__.CREATE_SHAPE[1] - 60), 0, thickness=2)
        return img

    def distorse(img, function=None, axis='x', alpha=1.0, beta=1.0):
        # can use with color or gray scale image
        # example code

        # x directional distorsion
        # img = IP.distorse(img,function='sine',axis='x',alpha=20,beta=2)

        # y directional distorsion
        # img = IP.distorse(img,function='sine',axis='y',alpha=20,beta=2)

        # both x and y directional distorsion
        # img = IP.distorse(img,function='sine',axis='x',alpha=20,beta=2)
        # img = IP.distorse(img,function='sine',axis='y',alpha=20,beta=2)

        # function are 'sine', 'linear' and 'inverse linear'

        if function != None:
            if function == 'sine':
                A = img.shape[0] / alpha
                w = beta / img.shape[1]

                dist_func = lambda x: A * np.sin(2.0 * np.pi * x * w)
            elif function == 'linear':
                dist_func = lambda x: alpha * x + beta
            elif function == 'inv_linear':
                dist_func = lambda x: -alpha * x - beta
            if axis == 'x':
                for i in range(img.shape[1]):
                    img[:, i] = np.roll(img[:, i], int(dist_func(i)))
            elif axis == 'y':
                for i in range(img.shape[0]):
                    img[i, :] = np.roll(img[i, :], int(dist_func(i)))
        return img

    def magnifly(image, percentage=100, shiftxy=[0, 0]):
        # can use with color or gray scale image
        # example code
        # img = IP.magnifly(img,150,shiftxy=[-30,-50])

        # percentage control how big/small the output image is.
        # shiftxy is with respect to top left conner

        try:
            x, y, c = image.shape
        except:
            x, y = image.shape
        x_ = x * percentage // 100
        y_ = y * percentage // 100

        img = cv2.resize(image, (y_, x_),interpolation=cv2.INTER_LANCZOS4)
        base = np.ones((x, y)) * 255.0
        base = Image.fromarray(base)

        img = Image.fromarray(img)
        base.paste(img, (-(x_ - x) // 2 + shiftxy[0], -(y_ - y) // 2 + shiftxy[1]))
        fuck=np.array(base, dtype=np.uint8)
        # base.show('hello')
        return np.array(base, dtype=np.uint8)

    '''IP = Image_Processing_And_Do_something_to_make_Dataset_be_Ready()
        img = cv2.imread('one.jpg',0)
        img = IP.distorse(img,function='sine',axis='x',alpha=20,beta=2)
        img = IP.distorse(img,function='sine',axis='y',alpha=20,beta=2)
        cv2.imshow('img',img)
        cv2.waitKey(0)'''

    class Plate():
        #A class for plate
        def __init__(self, image, cnt, word_cnt,extract_shape):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            word_rect = cv2.minAreaRect(word_cnt)
            word_box = cv2.boxPoints(word_rect)
            word_box = np.int0(word_box)

            # print(word_rect)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(color, [box], 0, (0, 0, 255), 2)
            cv2.drawContours(color, [word_box], 0, (0, 255, 0), 2)

            matrix = cv2.getRotationMatrix2D((cx, cy), rect[2], 1)
            # self.UnrotateWord = Image_Processing_And_Do_something_to_make_Dataset_be_Ready.remove_perspective(image,
            #                                                                                                   word_box,
            #                                                                                                 (50, 25))
            # print(matrix)
            self.image = image
            self.cnt = cnt
            self.PlateBox = box
            self.Original_Word_Size = word_rect[1]
            self.WordBox = word_box
            self.CenterPlate = [cx, cy]
            self.angle = rect[2]
            # print(word_rect)
            self.word_angle = word_rect[2]
            self.show = color
            self.UnrotateImg = cv2.warpAffine(image, matrix, image.shape, borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=255)
            if word_rect[1][0] > word_rect[1][1]:
                y1 = int(word_rect[1][0] / 2) + int(word_rect[0][0])
                y2 = int(word_rect[0][0]) - int(word_rect[1][0] / 2)
                x1 = int(word_rect[1][0] / 2) + int(word_rect[0][1])
                x2 = int(word_rect[0][1]) - int(word_rect[1][0] / 2)
            else:
                y1 = int(word_rect[1][1] / 2) + int(word_rect[0][0])
                y2 = int(word_rect[0][0]) - int(word_rect[1][1] / 2)
                x1 = int(word_rect[1][1] / 2) + int(word_rect[0][1])
                x2 = int(word_rect[0][1]) - int(word_rect[1][1] / 2)
            # print([x2,x1,y2,y1])
            # print(cx,cy)
            self.UnrotateWord = cv2.resize(self.UnrotateImg[x2:x1, y2:y1], extract_shape)

    def get_plate(image,extract_shape):
        image1 = 255 - image
        image1 = __class__.morph(image1, __class__.DILATE, [15, 15])
        img, contours, hierarchy = cv2.findContours(image1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        all_plate = []
        for cnt, hier, i in zip(contours, hierarchy[0], range(3)):
            if hier[3] == -1:
                hierach = hier
                index = 0
                while hierach[2] != -1:
                    index = hierach[2]
                    hierach = hierarchy[0, index]
                all_plate.append(__class__.Plate(image, cnt, contours[index],extract_shape))
        return all_plate


    # example
    # img = cv2.imread('ThreeEN.jpg',0)
    # cv2.imshow('org',img)
    # img = Image_Processing_And_Do_something_to_make_Dataset_be_Ready.ztretch(img,axis='horizontal',percentage=0.6)
    # cv2.imshow('result',img)
    # cv2.waitKey(0)

    def ztretch(image,percentage=1.0,axis='horizontal'):
        y,x = image.shape
        if axis == 'horizontal':

            x_ = int(x*percentage)
            y_ = y
        elif axis == 'vertical':
            x_ = x
            y_ = int(y*percentage)
        image = cv2.resize(image,(x_,y_))
        if percentage >= 1.0:
            image = image[(y_//2)-(y//2):(y_//2)+(y//2),(x_//2)-(x//2):(x_//2)+(x//2)]
        else:
            base = np.ones((y,x), np.uint8)*255
            # print((y//2)-(y_//2),(y//2)+(y_//2),(x//2)-(x_//2),(x//2)+(x_//2))
            base[(y//2)-(y_//2):(y//2)-(y_//2)+(y_),(x//2)-(x_//2):(x//2)-(x_//2)+x_] = image
            image = base
        return image

    def Adapt_Image(image):
        inv_image = 255-image
        dilate = cv2.dilate(inv_image,np.ones((10,10)))
        ret,cnt,hierarchy= cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        rect = cv2.minAreaRect(cnt[0])
        if rect[1][0] > rect[1][1]:
            y1 = int(rect[1][0] / 2) + int(rect[0][0])
            y2 = int(rect[0][0]) - int(rect[1][0] / 2)
            x1 = int(rect[1][0] / 2) + int(rect[0][1])
            x2 = int(rect[0][1]) - int(rect[1][0] / 2)
        else:
            y1 = int(rect[1][1] / 2) + int(rect[0][0])
            y2 = int(rect[0][0]) - int(rect[1][1] / 2)
            x1 = int(rect[1][1] / 2) + int(rect[0][1])
            x2 = int(rect[0][1]) - int(rect[1][1] / 2)
        return cv2.resize(image[x2:x1, y2:y1], (30,60))
    # extract plate from image
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.morph(img,ipaddr.DILATE,[15,15])
       cv2.imshow('img',img)
       cv2.waitKey(0)'''




