import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr


def Generate_Image_Data(font, fontsize=32, shape=(40, 40), borderthickness=3, translate=None, rotate=None, blur=None,
                        magnify=None, distort=None, savepath="", fontpath="", imageshow=False, detectblank=False,
                        allfont=False, word="ALL", specialcase=False):
    # Load all font in directory that is truetype or open type
    if allfont:
        font = [x for x in listdir(fontpath) if
                ".ttf" in x or ".otf" in x or ".ttc" in x or ".TTF" in x or ".OTF" in x or ".TTC" in x]

    wordlist = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9", "ศูนย์ ", "หนึ่ง ", "สอง ", "สาม ", "สี่ ", "ห้า ", "หก ", "เจ็ด ", "แปด ",
                "เก้า "]
    filename = {"zero": "zero", "one": "one", "two": "two", "three": "three", "four": "four", "five": "five",
                "six": "six", "seven": "seven", "eight": "eight", "nine": "nine", "0": "0", "1": "1", "2": "2",
                "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "ศูนย์ ": "ZeroTH",
                "หนึ่ง ": "OneTH", "สอง ": "TwoTH", "สาม ": "ThreeTH", "สี่ ": "FourTH", "ห้า ": "FiveTH",
                "หก ": "SixTH",
                "เจ็ด ": "SevenTH", "แปด ": "EightTH", "เก้า ": "NineTH"}
    # choose word to be created
    if word is "EN":
        wordlist = wordlist[0:10]
    elif word is "NUM":
        wordlist = wordlist[10:20]
    elif word is "TH":
        wordlist = wordlist[20:]
    # check variable type
    if type(font) is str:
        font = [font]
        # check process
    if rotate is None:
        rotate = [0]
    if blur is None:
        blur = [-1]
    else:
        blur.append(-1)
    if translate == None:
        translate = [(0, 0)]
    if magnify == None:
        magnify = [100]
    if distort == None:
        distort = None
    piccount = len(wordlist) * len(rotate) * len(blur) * len(translate) * len(magnify)
    print("one font generate:   " + str(piccount))
    print("all font generate:   " + str(len(font) * piccount))
    # iterate through font
    for f in font:
        print(f)
        # iterate through number
        fontname = f.split('.')[0]
        for num in wordlist:
            # Constructing base image
            image = ipaddr.font_to_image(fontpath + f, 32, 0, num)
            # extract word
            plate = ipaddr.get_plate(image, shape)
            extracted_word = plate[0].UnrotateWord
            if specialcase:
                extracted_word = ipaddr.binarize(extracted_word, method=ipaddr.OTSU_THRESHOLDING)

                # ADAPTIVE_CONTRAST_THRESHOLDING,
                # value=[5, 3])
            # create file name
            savefilename = fontname + "_" + "None" + "_" + "Magnify100" + "_" + "None" + "_" + "0" + "_" + "None" + "_" + "tran0l0""_" + \
                           filename[num] + ".jpg"
            cv2.imwrite(savepath + savefilename, extracted_word)

            # Show base image
            skip = False
            if imageshow and not skip:
                cv2.imshow("base", extracted_word)
                key = cv2.waitKey(0)
                if key == ord('s'):
                    skip = True

            # iterate through angle
            for angle in rotate:
                # rotate image

                rotate_image = ipaddr.rotation(extracted_word, (shape[0] / 2, shape[1] / 2), angle)
                rotate_image = ipaddr.binarize(rotate_image, method=ipaddr.ADAPTIVE_CONTRAST_THRESHOLDING,
                                             value=[3, 3])
                ret,rotate_image = cv2.threshold(rotate_image, 127, 255, cv2.THRESH_BINARY)
                # construct file name and save
                savefilename = fontname + "_" + "None" + "_" + "Magnify100" + "_" + "None" + "_" + str(
                    angle) + "_" + "None" + "_" + "trans0l0""_" + filename[num] + ".jpg"
                cv2.imwrite(savepath + savefilename, rotate_image)

                if imageshow and not skip:
                    cv2.imshow("base", rotate_image)
                    key = cv2.waitKey(0)
                    if key == ord('s'):
                        skip = True

                # iterate through blur method
                for blurmedthod in blur:
                    # blur image
                    if blurmedthod == 11:
                        blur_image = ipaddr.blur(rotate_image, blurmedthod, [3, 200, 200])
                        blurcode = "blur11"
                    elif blurmedthod == -1:
                        blur_image = rotate_image
                        blurcode = "None"
                    else:
                        blur_image = ipaddr.blur(rotate_image, blurmedthod, 3)
                        blurcode = "blur" + str(blurmedthod)

                    if not specialcase:
                        blur_image = ipaddr.binarize(blur_image, method=ipaddr.ADAPTIVE_CONTRAST_THRESHOLDING,
                                                     value=[3,3])
                        ret,blur_image = cv2.threshold(blur_image, 127, 255, cv2.THRESH_BINARY)
                    else:
                        blur_image = ipaddr.binarize(blur_image, method=ipaddr.ADAPTIVE_CONTRAST_THRESHOLDING,
                                                     value=[3, 3])
                        blur_image = ipaddr.binarize(blur_image, method=ipaddr.OTSU_THRESHOLDING)

                    # blur_image = ipaddr.remove_noise(blur_image, 8, 3)
                    # construct file name and save
                    savefilename = fontname + "_" + str(blurcode) + "_" + "Magnify100" + "_" + "None" + "_" + str(
                        angle) + "_" + "None" + "_" + "trans0l0""_" + filename[num] + ".jpg"

                    cv2.imwrite(savepath + savefilename, blur_image)

                    if imageshow and not skip:
                        cv2.imshow("base", blur_image)
                        key = cv2.waitKey(0)
                        if key == ord('s'):
                            skip = True
                    # iterate through magnify value
                    for magnifyratio in magnify:
                        magnify_image = ipaddr.magnifly(blur_image, magnifyratio)
                        ret, magnify_image= cv2.threshold(magnify_image, 127, 255, cv2.THRESH_BINARY)
                        savefilename = fontname + "_" + str(blurcode) + "_" + "Magnify" + str(
                            magnifyratio) + "_" + "None" + "_" + str(
                            angle) + "_" + "None" + "_" + "trans0l0""_" + filename[num] + ".jpg"
                        cv2.imwrite(savepath + savefilename, magnify_image)
                        if imageshow and not skip:
                            cv2.imshow("base", magnify_image)
                            key = cv2.waitKey(0)
                            if key == ord('s'):
                                skip = True
    return 0
