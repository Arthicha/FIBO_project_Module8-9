import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
import random

np.set_printoptions(threshold=np.inf)
def Generate_Image_Data(font, fontsize=32, shape=(40, 40), borderthickness=3, translate=None, rotate=None,
                        rotation_bound=[45, -45], blur=None, magnify=0, magnify_bound=[101, 90], stretch=0,
                        stretch_bound=[1.11, 0.9], distort=None, savepath="", fontpath="",
                        imageshow=False, detectblank=False,
                        allfont=False, word="ALL", save=True):
    if allfont:
        font = [x for x in listdir(fontpath) if
                ".ttf" in x or ".otf" in x or ".ttc" in x or ".TTF" in x or ".OTF" in x or ".TTC" in x]
        font =font[:-1]
        random.shuffle(font)
    wordlist = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9", "ศูนย์ ", "หนึ่ง ", "สอง ", "สาม ", "สี่ ", "ห้า ", "หก ", "เจ็ด ", "แปด ",
                "เก้า "]
    filename = {"zero": "zero", "one": "one", "two": "two", "three": "three", "four": "four", "five": "five",
                "six": "six", "seven": "seven", "eight": "eight", "nine": "nine", "0": "0", "1": "1", "2": "2",
                "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "ศูนย์ ": "ZeroTH",
                "หนึ่ง ": "OneTH", "สอง ": "TwoTH", "สาม ": "ThreeTH", "สี่ ": "FourTH", "ห้า ": "FiveTH",
                "หก ": "SixTH",
                "เจ็ด ": "SevenTH", "แปด ": "EightTH", "เก้า ": "NineTH"}

    # print(list(range(stretch_bound[1],stretch_bound[0],0.05)))




    if word is "EN":
        wordlist = wordlist[0:10]
    elif word is "NUM":
        wordlist = wordlist[10:20]
    elif word is "TH":
        wordlist = wordlist[20:]

    multiplicate = 0
    if translate is None:
        translate_quantify = 1
    else:
        translate_quantify = len(translate)
        multiplicate = 1
    if rotate is 0:
        rotate_quantify = 1
    else:
        rotate_quantify = rotate
        multiplicate = 1
    if blur is None:
        blur_quantify = 1
    else:
        blur_quantify = len(blur)
        multiplicate = 1
    if magnify is 0:
        magnify_quantify = 1
    else:
        magnify_quantify = magnify
        multiplicate = 1
    if stretch is 0:
        stretch_quantify = 1
    else:
        stretch_quantify = stretch*2
        multiplicate = 1

    # print(len(font))
    # print([x for x in range(0,magnify_quantify) ])
    # print([x for x in range(0, stretch_quantify//2)])
    piccount = len(font)+len(
        font) * multiplicate * translate_quantify * rotate_quantify * blur_quantify * magnify_quantify *( stretch_quantify+1)
    print("pic per num : " + str(piccount))
    for x in wordlist:
        write = ""
        skip = False
        n=0
        for y in font:
            print(y)
            img = ipaddr.font_to_image(fontpath + y, fontsize, 0, x)
            # cv2.imshow("suck",img)
            plate = ipaddr.get_plate(img, shape)
            extracted_word = plate[0].UnrotateWord
            # extracted_word=255-extracted_word
            # extracted_word = ipaddr.binarize(extracted_word, method=ipaddr.SAUVOLA_THRESHOLDING, value=29)
            extracted_word = ipaddr.binarize(extracted_word, method=ipaddr.SAUVOLA_THRESHOLDING,value=29)
            if imageshow and not skip:
                cv2.imshow("original", extracted_word)
                key = cv2.waitKey(0)
                if key == ord('s'):
                    skip = True
            extracted_word_string = (extracted_word.ravel()) / 255
            extracted_word_string = np.array2string(extracted_word_string.astype(int),max_line_width=80000, separator=',')
            n+=1
            write += extracted_word_string[1:-1] + "\n"
            for z in range(0, magnify_quantify):
                if magnify == 0:
                    magnify_img = extracted_word
                else:
                    magnify_img = ipaddr.magnifly(extracted_word,
                                                  percentage=random.randint(magnify_bound[1], magnify_bound[0]))
                    magnify_img=255-magnify_img
                    magnify_img = ipaddr.binarize(magnify_img, method=ipaddr.ADAPTIVE_CONTRAST_THRESHOLDING,value=[15,-0.8])
                    magnify_img=255-magnify_img

                    magnify_img_string = np.array2string(((magnify_img.ravel()) / 255).astype(int), max_line_width=80000,
                                                         separator=',')
                    n+=1
                    write += magnify_img_string[1:-1] + "\n"
                if imageshow and not skip:
                    cv2.imshow("magnify", magnify_img)
                    key = cv2.waitKey(0)
                    if key == ord('s'):
                        skip = True
                for k in range(0, stretch_quantify//2):
                        if stretch == 0:
                            stretch_img = magnify_img
                        else:
                            stretch_img = ipaddr.ztretch(magnify_img,
                                                         percentage=round(random.uniform(stretch_bound[1], stretch_bound[0]),2),
                                                         axis='horizontal')

                            if imageshow and not skip:
                                cv2.imshow("stretch", stretch_img)
                                key = cv2.waitKey(0)
                                if key == ord('s'):
                                    skip = True
                            stretch_img_string = np.array2string(((stretch_img.ravel()) / 255).astype(int), max_line_width=80000,
                                                                 separator=',')
                            write += stretch_img_string[1:-1] + "\n"
                            stretch_img = ipaddr.ztretch(magnify_img,
                                                         percentage=random.uniform(stretch_bound[1], stretch_bound[0]),
                                                         axis='vertical')
                            if imageshow and not skip:
                                cv2.imshow("stretch", stretch_img)
                                key = cv2.waitKey(0)
                                if key == ord('s'):
                                    skip = True
                            stretch_img_string = np.array2string(((stretch_img.ravel()) / 255).astype(int), max_line_width=80000,
                                                                 separator=',')
                            n+=2
                            write += stretch_img_string[1:-1] + "\n"
            if n==len(font)*0.2:
                print(n)
                if save:
                    open(savepath+"dataset" + "_" + filename[x]+"_"+"test" + '.txt', 'w').close()
                    file = open(savepath+"dataset"+"_"+filename[x] +"_"+"test"+ '.txt', 'a')
                    file.write(write)
                    file.close()
                    write = ""
            elif n == len(font)*0.4:
                print(n)
                if save:
                    open(savepath+"dataset" + "_" + filename[x] + "_" + "validate" + '.txt', 'w').close()
                    file = open(savepath+"dataset" + "_" + filename[x] + "_" + "validate" + '.txt', 'a')
                    file.write(write)
                    file.close()
                    write = ""
        if save:
            open(savepath+"dataset" + "_" + filename[x] + "_" + "train" + '.txt', 'w').close()
            file = open(savepath+"dataset" + "_" + filename[x] + "_" + "train"+ '.txt', 'a')
            file.write(write)
            file.close()
        print(filename[x])
        print(n)