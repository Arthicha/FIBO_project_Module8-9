from Dataset_GeneratorV2 import Generate_Image_Data as GID

# C:\\Users\cha45\Desktop\\fuc\TH_dataset_img3\\
# C:\\Users\cha45\Desktop\\fuc\TH_dataset_img\\"C:\\Users\cha45\Desktop\\fuc\ENG_dataset_img2\\
savePath = "C:\\Users\cha45\Desktop\\fuc\EN_dataset_img6\\"
# # C:\\Users\cha45\Downloads\\front thai\\front thai\\
# # C:\\Users\cha45\Downloads\DATA SET - fonts2-20180226T150229Z-001\DATA SET - fonts2\\
# # C:\\Users\cha45\Downloads\DATA SET - fonts-20180224T101407Z-001\DATA SET - fonts\\
Font_Path = "C:\\Users\cha45\Desktop\\fuc\ENGFONT\\"
RotateAngle = [x for x in range(15, -20, -5)]
RotateAngle2 = [x for x in range(20, -21, -10)]
blur_method = [9, 10, 11]
magnify = [x for x in range(90, 115, 5)]
distort = []

GID("", shape=(60, 30), translate=None, rotate=0,
    rotation_bound=[45, -45], blur=None, magnify=4, magnify_bound=[111, 90], stretch=1,
    stretch_bound=[1.11, 0.9], distort=None, savepath="", fontpath=Font_Path,
    imageshow=False, detectblank=False,
    allfont=True, word="NUM")
# GID("", rotate=RotateAngle, blur=blur_method, magnify=magnify, savepath=savePath, fontpath=Font_Path, imageshow=False,
#     allfont=True, word="NUM")
