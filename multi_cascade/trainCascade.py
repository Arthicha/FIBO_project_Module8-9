# ~/virtualenv/ROBOTICS_studios/bin/python


'''*************************************************
*                                                  *
*                  import library                  *
*                                                  *
*************************************************'''

import os
import sys
import platform
from random import choice, randrange, shuffle
from time import sleep

from PIL import Image, ImageOps
from math import sqrt, pow
import numpy as np


'''*************************************************
*                                                  *
*                 define valuable                  *
*                                                  *
*************************************************'''

dirCom = '/'
scaleWeightHeight = 0.5
scalePosNeg = 1
memoryUse = 8192

listOfClass = [0,1,2,3,4,5,6,7,8,9]+['zero','one','two','three','four','five','six',
                'seven','eight','nine']+['ZeroTH','OneTH','TwoTH','ThreeTH','FourTH','FiveTH','SixTH',
                'SevenTH','EightTH','NineTH']

'''*************************************************
*                                                  *
*             define anoymous function             *
*                                                  *
*************************************************'''

WHfromArray1D = lambda arraySize : ( int(sqrt(arraySize*scaleWeightHeight)), int(sqrt(arraySize/scaleWeightHeight)) )


'''*************************************************
*                                                  *
*                   main library                   *
*                                                  *
*************************************************'''

def main():
    global dirCom, weight, height, listOfClass 
    inputKey = sys.argv[1:]
    
    if platform.system() == 'Linux':
        dirCom = '/'
    elif platform.system() == 'Windows':
        dirCom = '\\'

    if inputKey == [] or str(inputKey[0]) == 'help':
        print('set data')
        print('trainCascade.py [method] [param] \nmethod:\tgen_image\tresize\t\t\tcreate_bg\tremove_xml')
        print('param:\tnumber/class\tmain_image -- size\tmain_class\n\t1000\t\ttrain-0 24\t\tone')
        print('------------------------------------------------------------')
        print('generate classification : required --> libopencv-dev, linux os ')
        print('trainCascade.py [method] [param]')
        print('method:\tcreatesamples\t\ttraincascade\t\t\t\t\thaartraining\tperformance')
        print('param:\tmain_class -- number\tmain_class -- numpos -- numneg -- numstate\tnon_finished\tnon_finished')
        print('\tone 1000\t\tone 800 2400 10\t\t\t\t\t-\t\t-')
        print('------------------------------------------------------------------------------------')
        print('generate 30 classification  : required --> libopencv-dev, linux os ')
        print('trainCascade.py autogen [param]')
        print('param:\tnumber/class -- main_image -- h_size -- numstate -- state(repackage,unrepackage) -- feature(HAAR, HOG, LBP)')
        print('\t1000 \t\ttrain-0 \t24 \t10 \t\trepackage \t\t\tHAAR\n')

    elif str(inputKey[0]) == 'remove_xml':
        deleteMainCascadeFile()
        
    elif str(inputKey[0]) == 'resize':
        try :
            resize_image(selectFile = (str(inputKey[1])+'.png'), size = int(inputKey[2]))

        except Exception :
            resize_image()

    elif str(inputKey[0]) == 'create_bg':
        try :
            create_bg_txt(select_value = str(inputKey[1]))
        except Exception as e:
            sys.exit('error argument : '+str(e))

    elif str(inputKey[0]) == 'gen_image':
        try :
            generate_picture(limitFilePerClass = int(inputKey[1]))
        except Exception as e:
            generate_picture()

    elif str(inputKey[0]) == 'createsamples' and platform.system() == 'Linux' :
        try:
            run_opencv_createsamples(main_class=str(inputKey[1]),number=str(inputKey[2]))
        except Exception as e:
            sys.exit('createsamples argument error : '+str(e))

    elif str(inputKey[0]) == 'traincascade' and platform.system() == 'Linux' :
        if str(inputKey[1]) in str(listOfClass) : 
            try:
                run_opencv_traincascade(main_class=str(inputKey[1]),numpos=str(inputKey[2]),numneg=str(inputKey[3]),numstate=str(inputKey[4]),feature=str(inputKey[5]))
            except Exception as e:
                sys.exit('traincascade argument error : '+str(e))
        else :
            sys.exit('out of class')

    elif str(inputKey[0]) == 'haartraining' and platform.system() == 'Linux' :
        sys.exit('this method not finished\nPlease run prepare_haarCascade.py help')

    elif str(inputKey[0]) == 'performance' and platform.system() == 'Linux' :
        sys.exit('this method not finished\nPlease run prepare_haarCascade.py help')

    elif str(inputKey[0]) == 'autogen' and platform.system() == 'Linux' :
        try :
            AutoGenerateClassification(numberPerClass=str(inputKey[1]), main_img=str(inputKey[2]), size=str(inputKey[3]), numstate=str(inputKey[4]), state=str(inputKey[5]), feature=str(inputKey[6]))
        except Exception as e:
            sys.exit('error:'+str(e))
    else :
        sys.exit("method doesn't have in program\n-----------------------\nPlease run prepare_haarCascade.py help")


'''*************************************************
*                                                  *
*                    sub module                    *
*                                                  *
*************************************************'''

def generate_picture(limitFilePerClass = 50):
    '''generate picture from file in folder dataCompress and save to folder
    dataExtract with limit picture per class.'''

    '''*************************************************
    *                                                  *
    *              config generate number              *
    *                                                  *
    *************************************************'''
    numCount = 0
    numKeep = 0



    '''*************************************************
    *                                                  *
    *                   prepare data                   *
    *                                                  *
    *************************************************'''

    suffix = ['test','train','validate']
    listOfClass = [0,1,2,3,4,5,6,7,8,9]+['zero','one','two','three','four','five','six',
                        'seven','eight','nine']+['ZeroTH','OneTH','TwoTH','ThreeTH','FourTH','FiveTH','SixTH',
                        'SevenTH','EightTH','NineTH']

    '''*************************************************
    *                                                  *
    *                   remove old data                *
    *                                                  *
    *************************************************'''
    try:
        fileList= [f for f in os.listdir('dataExtract')]
        for f in fileList:
            os.remove(os.path.join('dataExtract',f)) 

    except Exception:
        print("error to remove file in dataExtract folder")

    '''*************************************************
    *                                                  *
    *              read & generate data                *
    *                                                  *
    *************************************************'''

    for s in range(1,3): # traain & validate
        for j in range(0,30): # 30 class
            object = listOfClass[j]
            f = open('dataCompress'+dirCom+'dataset_'+str(object)+'_'+suffix[s]+'.txt','r')
            image = str(f.read()).split('\n')[:-1]
            f.close()

            numKeep += numCount
            numCount = 0
            for i in range(len(image)):
                
                path = 'dataExtract'+dirCom+str(object)+'_'+suffix[s]+'-'+str(numCount)+'.png'

                image[i] = np.fromstring(image[i], dtype=float, sep=',')
                image[i] = np.array(image[i])
                
                image[i] = np.reshape(image[i],WHfromArray1D(len(image[i])))
                img = Image.fromarray((image[i]*255).astype(np.uint8))
                # img = ImageOps.invert(img) 

                img.save(path)

                if numCount > int(limitFilePerClass)-1 :
                    break
                if (numCount%int(int(limitFilePerClass)/2)) == 0 :
                    print("generate"+str(numKeep+numCount)+ ":"+suffix[s]+'-'+str(object) +"-"+str(numCount))

                numCount+=1

def resize_image(selectFile = 'test-0.png', size = 24):
    '''resize image from folder dataExtract and save to folder data, 
        And select main image.'''
    
    print('select file *'+selectFile +" : " +str(size))        

    '''*************************************************
    *                                                  *
    *                   remove old data                *
    *                                                  *
    *************************************************'''
    try:
        fileList= [f for f in os.listdir('data')]
        for f in fileList:
            os.remove(os.path.join('data',f)) 
    except Exception:
        print("error to remove file in data folder")

    try:
        fileList= [f for f in os.listdir('main_img')]
        for f in fileList:
            os.remove(os.path.join('main_img',f)) 
    except Exception:
        print("error to remove file in main_img folder")

    '''*************************************************
    *                                                  *
    *            resize and select main image          *
    *                                                  *
    *************************************************'''
    
    path = 'dataExtract'
    fileList= [f for f in os.listdir(path)]

    for f in fileList:
        img = Image.open(path+dirCom+f)
        
        if img.height < int(size) or img.width < int(size):
            sys.exit('size is bigger than '+str(img.height)+','+str(img.width))
        
        img = img.resize((int(int(size)/scaleWeightHeight),int(size)),Image.ANTIALIAS)

        img.save('data'+dirCom+f)
        
        if f.split('_')[1] == selectFile:
            img.save('main_img'+dirCom+f)


def create_bg_txt(select_value):
    '''use image from dataExtract and input string to write bg_neg.txt and bg_pos.txt .'''
    
    '''*************************************************
    *                                                  *
    *            remove & create old file              *
    *                                                  *
    *************************************************'''

    if os.path.isfile('bg_pos.txt') :
        os.remove('bg_pos.txt')
    if os.path.isfile('bg_neg.txt') :
        os.remove('bg_neg.txt')

    f_pos = open("bg_pos.txt","w+")
    f_neg = open("bg_neg.txt","w+")
    
    '''*************************************************
    *                                                  *
    *                 random data list                 *
    *                                                  *
    *************************************************'''
    
    listData = os.listdir('data')
    randomList = []
    while len(listData) > 0 :
        randomData = choice(listData)
        randomList.append(randomData)
        listData.remove(randomData)    
 
    '''*************************************************
    *                                                  *
    *            split positive and negative           *
    *                                                  *
    *************************************************'''


    countPos =0
    countNeg =0
    if str(select_value) in str(listOfClass):
        for f in randomList:
            if str(f.split('_')[0]) == str(select_value):
                f_pos.write("data"+dirCom+f+"\n")
                countPos+=1
            # else:
            #     f_neg.write("data"+dirCom+f+"\n")
            #     countNeg+=1
        countNegs = int(countPos/len(listOfClass))*len(listOfClass) * scalePosNeg
        
        keepList = []
        while (countNeg < countNegs):
            
            key = str(randrange(0,len( [i for i in [ str(j).split('0_train') for j in randomList ] if len(i) == 2] )))+'.png'
            
            for selectClass in listOfClass:
                keepList.append("data"+dirCom+str(selectClass)+'_train-'+str(key)+"\n")
                countNeg+=1 

        shuffle(keepList)
        for selectList in keepList:
            f_neg.write(selectList)
                   
    


    else:
        sys.exit('out of class')
    
    print("number of positive : "+str(countPos))
    print("number of negative : "+str(countNeg))



def run_opencv_createsamples(main_class='',number=''):
    ''' opencv_createsamples library from libopencv-dev .\n
        To generate vector file for run opencv_traincascade .'''

    if main_class=='' or number=='':
        sys.exit('main class or number is invalid')

    weight, height = Image.open('main_img'+dirCom+os.listdir('main_img')[0]).size

    command = 'opencv_createsamples -img main_img'+dirCom+str(main_class)+'* -bg bg_pos.txt -vec positives.vec -bgcolor 0 -maxxangle 1.2 -maxyangle 1.2 -maxzangle 0.5 -num '+str(number) +' -w '+str(weight)+' -h '+str(height)
    os.system(command)

def run_opencv_traincascade(main_class='0',numpos=0,numneg=0,numstate=0,feature='HAAR'):
    ''' opencv_traincascade library from libopencv-dev .\n
        To generate haarCascade classification file. '''

    if numpos==0 or numneg==0 or numstate==0 :
        sys.exit('numpos | numneg | numstate is 0')

    file_0 = os.listdir('main_img')[0]
    
    weight, height = Image.open('main_img'+dirCom+os.listdir('main_img')[0]).size
    
    command = 'opencv_traincascade -featureType '+str(feature)+' -data output_data'+dirCom+str(main_class) +dirCom +' -vec positives.vec -bg bg_neg.txt -numPos '+str(numpos)+' -numNeg '+str(numneg)+' -numStages '+str(numstate)+' -w '+str(weight)+' -h '+str(height)+' -precalcValBufSize '+str(memoryUse)+' -precalcIdxBufSize '+str(memoryUse)
    os.system(command)

def run_opencv_haartraining():
    '''Now, don't know how it use.'''
    
    pass

def run_opencv_performance():
    '''Now, don't know how it use.'''
        
    pass

def AutoGenerateClassification(numberPerClass=1000, main_img='train-0',size=24, numstate=10, state ='repackage', feature='HAAR'):
    '''auto generate 30 classification by auto parameter.'''
    
    if str(state) == 'repackage' :
        print('gen_image '+str(numberPerClass)+' per class')
        generate_picture(limitFilePerClass = numberPerClass)
        print('done')
        resize_image(selectFile=str(main_img)+'.png',size=size)
    
    for selectClass in listOfClass:
        create_bg_txt(select_value=selectClass)
        
        with open('bg_neg.txt','r') as f :
            countNeg = len(str(f.read()).split('\n'))
        with open('bg_pos.txt','r') as f :
            countPos = len(str(f.read()).split('\n'))

        num = predictNumPosNumNeg(countPos=countPos*4/5,countNeg=countNeg*4/5)    
        # renum = predictNumPosNumNeg(countPos=num[0]*4/5,countNeg=num[1]*4/5)    
        run_opencv_createsamples(main_class=selectClass,number=int(num[0]*5/4))
        run_opencv_traincascade(main_class=selectClass,numpos=int(num[0]),numneg=int(num[1]),numstate=int(numstate),feature=feature)

    print('wait for delete unuse data')
    for f in [f for f in os.listdir('dataExtract')]:
        os.remove(os.path.join('dataExtract',f)) 
    for f in [f for f in os.listdir('data')]:
        os.remove(os.path.join('data',f)) 
    print('remove extract data')
    for f in [f for f in os.listdir('main_img')]:
        os.remove(os.path.join('main_img',f)) 
    print('remove main_img')


        

def deleteMainCascadeFile():
    '''delete all cascade file in folder output_data. '''
    print('removing all main cascade file in folder output_data')
    for selectClass in listOfClass :
        for f in [i for i in os.listdir('output_data'+dirCom+str(selectClass))] :
            os.remove(os.path.join('output_data'+dirCom+str(selectClass),f))
    return 0

def predictNumPosNumNeg(countPos,countNeg):
    ''' find NumPos and NumNeg in term i*pow(10,n) .'''
    countKeep = 0
    pos = int(countPos)
    neg = int(countNeg)
    while pos >= 10:
        pos /= 10
        countKeep+=1
    pos = int(pow(10,countKeep)*int(pos))

    countKeep = 0
    while neg >= 10:
        neg /= 10
        countKeep+=1
    neg = int(pow(10,countKeep)*int(neg))
    

    return [pos,neg]

if __name__ == '__main__':
    main()
