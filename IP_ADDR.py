__author__ = ['Zumo', 'Tew', 'Wisa']
__version__ = 1.10
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
from Foundation.Plate import Plate


class Image_Processing_And_Do_something_to_make_Dataset_be_Ready():
    # def __init__(self):
    # GLOBAL
    SHAPE = (640, 480)
    CREATE_SHAPE = (320, 240)

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

    def binarize(image, method = OTSU_THRESHOLDING, value=None):
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
        return img

    def remove_noise(image, method = KUWAHARA, value=5):
        if method == __class__.KUWAHARA:
            img = Filter.Filter_Kuwahara(image, value)
        elif method == __class__.WIENER:
            img = Filter.Filter_weiner(image, window_size=value[0], iteration=value[1])
        elif method == __class__.MEDIAN:
            img = Filter.Filter_median(image, window=value)
        else:
            sys.exit("Unknown method\n")
        return img

    def resize(image, method=INTER_LINEAR):
        img = cv2.resize(image, __class__.SHAPE, interpolation=method)
        return img

    def get_plate(image):
        img, contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        all_plate = []
        for cnt, hier, i in zip(contours, hierarchy[0], range(3)):
            if hier[3] == -1:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                hierach = hier
                index = 0
                while hierach[2] != -1:
                    index = hierach[2]
                    hierach = hierarchy[0, index]
                word_rect = cv2.minAreaRect(contours[index])
                word_box = cv2.boxPoints(word_rect)
                word_box = np.int0(word_box)
                all_plate.append(Plate(box, word_box))
        return all_plate

    def capture(cam, size=SHAPE, mode=GRAY_SCALE, config=None):
        ret, image = cam.read()
        if mode != __class__.BGR:
            image = cv2.cvtColor(image, mode, config)
        image = cv2.resize(image, size)
        return image

    def translate(image, value, config=None):
        matrix = np.float32([[1, 0, value[0]], [0, 1, value[1]]])
        if config is None:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=__class__.SHAPE)
        elif config[1] == __class__.BORDER_CONSTANT:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=__class__.SHAPE, flags=config[0],
                                 borderMode=config[1], borderValue=config[2])
        else:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=__class__.SHAPE, flags=config[0],
                                 borderMode=config[1])
        return img

    def blur(image, method=AVERAGING, value=5):
        if method == __class__.MEDIAN:
            img = cv2.medianBlur(image, value)
        elif method == __class__.AVERAGING:
            averaging_kernel = np.ones(value, value) / pow(value, 2)
            img = cv2.filter2D(image, -1, averaging_kernel)
        elif method == __class__.GAUSSIAN:
            img = cv2.GaussianBlur(image, value, 0)
        elif method == __class__.BILATERAL:
            img = cv2.bilateralFilter(image, value[0], value[1], value[2])
        else:
            sys.exit("Unknown method\n")
        return img

    def morph(image, mode=DILATE, value=[3,3]):
        matrix = np.ones((value[0], value[1]), np.float32)
        img = cv2.morphologyEx(image, mode, matrix)
        return img

    def remove_perspective(image, region, shape):
        pts1 = region
        pts2 = np.float32([0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(image, matrix, shape)
        return img

    def rotation(image, center_of_rotation, angle):
        matrix = cv2.getRotationMatrix2D((center_of_rotation[0], center_of_rotation[1]), angle, 1)
        #print(matrix)
        img = cv2.warpAffine(image, matrix, __class__.CREATE_SHAPE, borderMode=__class__.BORDER_CONSTANT, borderValue=255)
        return img

    def font_to_image(font, size=CREATE_SHAPE, index=0, string="0"):
        # Example
        # from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
        # image = ipaddr.font_to_image("angsana.ttc", 10, 0, "หนึ่ง")
        # cv2.imshow("one", image)
        # cv2.waitKey(0)

        Text_Font = ImageFont.truetype(font, size, index)
        w, h = Text_Font.getsize(string)
        img = Image.new("L", __class__.CREATE_SHAPE, color=255)
        image = ImageDraw.Draw(img)
        image.text(((__class__.CREATE_SHAPE[0] - w) / 2, (__class__.CREATE_SHAPE[1] - h) / 2), string, font=Text_Font,
                   fill="black")
        img = np.array(img)
        cv2.rectangle(img, (60, 20), (__class__.CREATE_SHAPE[0] - 60, __class__.CREATE_SHAPE[1] - 20), 0, thickness=2)
        return img

    def distorse( img, function=None, axis='x', alpha=1.0, beta=1.0):
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

    def magnifly( image, percentage=100, shiftxy=[0, 0]):
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

        img = cv2.resize(image, (x_, y_))
        base = np.ones((x, y)) * 255
        base = Image.fromarray(base)

        img = Image.fromarray(img)
        base.paste(img, (-(x_ - x) // 2 + shiftxy[0], -(y_ - y) // 2 + shiftxy[1]))
        # base.show('hello')
        return np.array(base, dtype=np.uint8)

    '''IP = Image_Processing_And_Do_something_to_make_Dataset_be_Ready()
    img = cv2.imread('one.jpg',0)
    img = IP.distorse(img,function='sine',axis='x',alpha=20,beta=2)
    img = IP.distorse(img,function='sine',axis='y',alpha=20,beta=2)
    cv2.imshow('img',img)
    cv2.waitKey(0)'''
