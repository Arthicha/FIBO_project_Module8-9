import numpy as np
import cv2
import os

def compressData(TextFileName,FilesLocation,ImageIndex,NumGroup='',functionIndex=0,Train=0,Error=0,functionIndex1=0,Train1=0,Error1=0,functionIndex2=0,Train2=0,Error2=0,TrainTrans=''):
    open(str(TextFileName)+'_test.txt', 'w').close()
    open(str(TextFileName)+'_train.txt', 'w').close()
    open(str(TextFileName)+'_validate.txt', 'w').close()
    path = FilesLocation
    dirs = os.listdir(path)
    numcount = 0
    datas = np.array([[]])

    def GetimgSiz():
            imgSam = cv2.imread(ImageIndex+str(dirs[0]),0)
            imsizSam = np.shape(imgSam)
            return (int(imsizSam[0])*int(imsizSam[1]))

    for files in dirs:
       a = files.split('_')
       d = a[len(a)-1]
       s = d.split(".")
       Num = s[0]
       ImgFunc = int(a[len(a)-functionIndex])
       ImgFunc1 = int(a[len(a)-functionIndex1])
       ImgFunc2 = int(a[len(a)-functionIndex2])
       ImgFuncTrans = str(a[len(a)-2])
       if Num == NumGroup:
           if (Train-Error <= ImgFunc <= Train+Error) and (ImgFuncTrans == TrainTrans) and (Train1-Error1 <= ImgFunc1 <= Train1+Error1) and (Train2-Error2 <= ImgFunc2 <= Train2+Error2):
               numcount+=1
               # print(files)
               img = cv2.imread(ImageIndex+str(files),0)
               ret,img = cv2.threshold(img,127,1,cv2.THRESH_BINARY)
               datas = np.append(datas, img)
               # print(numcount)
    # print(np.shape(datas))
    print(numcount)
    col = GetimgSiz()
    reshape = datas.reshape(numcount,col)
    np.random.shuffle(reshape)
    shape = np.shape(reshape)
    First20 = int(shape[0]*0.2)
    Mid60 = int(shape[0]*0.6)
    Last20 = First20+Mid60

    file = open(str(TextFileName)+'_test.txt','a')
    for k in range(0,First20):
        for m in range(shape[1]):
            num = str(int(float(np.array2string(reshape[k][m]))))
            file.write(num)
            file.write(',')
        file.write('\n')
    file.close()

    file = open(str(TextFileName)+'_train.txt','a')
    for k in range(First20+1,Last20):
        for m in range(shape[1]):
            num = str(int(float(np.array2string(reshape[k][m]))))
            file.write(num)
            file.write(',')
        file.write('\n')
    file.close()

    file = open(str(TextFileName)+'_validate.txt','a')
    for k in range(Last20,shape[0]):
        for m in range(shape[1]):
            num = str(int(float(np.array2string(reshape[k][m]))))
            file.write(num)
            file.write(',')
        file.write('\n')
    file.close()
# compressData('dataset_'+str(0)+'_r0','ENG_dataset_img','C:\\Users\\ncom220617\\PycharmProjects\\anaconda\\ENG_dataset_img\\',NumGroup=str(0),functionIndex=4,Train=0,Error=0,functionIndex1=2,TrainTrans='trans0l0')

for i in range(0,10):
    compressData('dataset_'+str(i)+'_r0','ENG_dataset_img','C:\\Users\\ncom220617\\PycharmProjects\\anaconda\\ENG_dataset_img\\',NumGroup=str(i),functionIndex=4,Train=0,Error=0,functionIndex1=6,Train1=0,Error1=0,functionIndex2=7,Train2=0,Error2=0,TrainTrans='trans0l0')
    print('ok'+str(i))

