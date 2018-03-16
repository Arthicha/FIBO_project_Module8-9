
__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = '2.0.0'


'''*************************************************
*                                                  *
*                  import module                   *
*                                                  *
*************************************************'''

# import the tensorflow

import tensorflow as tf
from tensorflow.python.framework import ops

# mathematical module
import numpy as np
import math
import random as ran
import random

# display module
import matplotlib.pyplot as plt

# system module
import os
import sys
import cv2
import copy

# my own library
from Tenzor import TenzorCNN,TenzorNN,TenzorAE

from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP



'''*************************************************
*                                                  *
*                 config tensorflow                *
*                                                  *
*************************************************'''

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# parallel operation
CPU_NUM = 0




'''*************************************************
*                                                  *
*                 configuration                    *
*                                                  *
*************************************************'''

# define augmentation mode
AUG_NONE = 0
AUG_DTSX = 1
AUG_DTSY = 2
AUG_DTSB = 3
AUG_LINX = 4
AUG_LINY = 5
AUG_LINB = 6
AUG_VALUE = [20,3]

PROG_PERF = 1
PROG_TEST = 2

PROGRAM = PROG_TEST

# dataset from 'mnist' or 'project'
DATA = 'PROJECT'

# choose machine learning model 'LATENT' or 'CNN'
model = 'CNN'

# continue previous model
CONTINUE = False

# save and get path
GETT_PATH = "D:\\2560\\FRA361_Robot_Studio\\FIBO_project_Module8-9\\model"
SAVE_PATH = "D:\\2560\\FRA361_Robot_Studio\\FIBO_project_Module8-9\\model"


BATCH2PRINT = 20
EPOCH = 2000
AUGMENT = AUG_NONE
DIVIDEY = 1
VALIDATE_SECTION = 100

# learning algorithm 'ADAM' or 'GRAD'
LEARNING_ALGORITHM = 'ADAM'


# debugging tool
TENSOR_BOARD = False
SHOW_AUG = False




'''*************************************************
*                                                  *
*                hyper parameter                   *
*                                                  *
*************************************************'''

if DATA is 'MNIST':
    imgSize = [28,28] # size of image
    N_CLASS = 10
elif DATA is 'PROJECT':
    imgSize = [30,60]
    N_CLASS = 30

CNN_HIDDEN_LAYER = [32,64,128] #amount of layer > 3
NN_HIDDEN_LAYER = [1,1]
AE_HIDDEN_LAYER = [imgSize[0]*imgSize[1],100,50,3,50,100,imgSize[0]*imgSize[1]]
KERNEL_SIZE = [[3,6],[3,6]]
POOL_SIZE = [[2,2],[3,3]]
STRIDE_SIZE = [2,3]

BATCH_SIZE = 2000

LEARNING_RATE = 0.001
KEEP_PROB = 0.9
MULTI_COLUMN = 1

'''*************************************************
*                                                  *
*                   function                       *
*                                                  *
*************************************************'''

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
                    img_r[:,0:5] = 255
                    img_r[:,60-6:60-1] = 255
                    img_r[0:5,:] = 255
                    img_r[30-6:30-1,:] = 255
                    subImg.append(img_r)
                    '''chkO = checkOreantation(img_r)
                    diff[a] = [chkO,copy.deepcopy(img_r)]
                if diff[0][0] > diff[1][0]:
                    subImg.append(diff[0][1])
                else:
                    subImg.append(diff[1][1])'''
                cv2.drawContours(org,[approx],0,(0,0,255),3)

    return org,subImg

'''*************************************************
*                                                  *
*                 data preparation                 *
*                                                  *
*************************************************'''


# Read dataset and divide them in to 3 group
#       1. testing set 20 percent -> just test
#       2. training set 60 percent -> train & learn
#       3. validation set 20 percent -> hyper-parameter tuning and model selection
# each group store in a list of [list of image ([img1,img2]), label (class)]

TestTrainValidate = [[],[],[]]
LabelTTT = [[],[],[]]
if DATA is 'MNIST':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
elif DATA is 'PROJECT':
    suffix = ['test','train','validate']
    listOfClass = [0,1,2,3,4,5,6,7,8,9]+['zero','one','two','three','four','five','six',
                       'seven','eight','nine']+['ZeroTH','OneTH','TwoTH','ThreeTH','FourTH','FiveTH','SixTH',
                       'SevenTH','EightTH','NineTH']
    for s in range(2,3):
        print('STATUS: process data',str(100.0*s/3.0))
        for j in range(0,N_CLASS):
            object = listOfClass[j]
            f = open('data0-9compress\\dataset_'+str(object)+'_all_'+suffix[s]+'.txt','r')
            image = str(f.read()).split('\n')[:100]
            f.close()
            delList = []
            for i in range(len(image)):
                image[i] = np.fromstring(image[i], dtype=float, sep=',')
                image[i] = np.array(image[i])
                image[i] = np.reshape(image[i],(60*30))
            TestTrainValidate[s] += image
            obj = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            obj[j] = 1
            LabelTTT[s] += np.full((len(image),N_CLASS),copy.deepcopy(obj)).tolist()
        if s == 1:
            if DATA is 'PROJECT':
                print('STATUS: shuffle-ing')
                def shuffly(a,b):
                    c = list(zip(a, b))
                    random.shuffle(c)
                    a, b = zip(*c)
                    return a,b
                a,b = shuffly(TestTrainValidate[1],LabelTTT[1])
                trainingSet = [np.array(a),np.array(b)]
                print('STATIS: complete shuffle-ing')
        del image
        del object
if DATA is 'MNIST':
    testingSet  = [mnist.test.images,mnist.test.labels]
    trainingSet = [mnist.train.images,mnist.train.labels]
    validationSet = [mnist.validation.images,mnist.validation.labels]
elif DATA is 'PROJECT':
    testingSet  = [TestTrainValidate[2],LabelTTT[2]]
    validationSet = [TestTrainValidate[2],LabelTTT[2]]

def randint(min,max):
    return ran.randint(min,max)

'''*************************************************
*                                                  *
*                 main function                    *
*                                                  *
*************************************************'''
def main(model='CNN',aug=0,value=None,GETT_PATH = None,SAVE_PATH=None,MAIN_HIDDEN_LAYER = [],NN_HIDDEN_LAYER = [],
         BATCH_SIZE = 16,BATCH2PRINT = 1000,EPOCH = 1,LEARNING_RATE = 0.01,KEEP_PROB = 0.9):


    global CNN_MODEL,LATENT_MODEL,LEARNING_ALGORITHM
    fin_AE = False
    if model is 'CNN':
        pass
    elif model is 'LATENT':
        pass
    else:
        sys.stderr.write('MODEL ERROR: '+ str(model))
        sys.exit(-1)

    #create interactive session
    sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=CPU_NUM,intra_op_parallelism_threads=CPU_NUM))
    if not fin_AE:
        #input and output as the placeholder
        with tf.name_scope('input_placeholder'):
            x = tf.placeholder(tf.float32, shape=[None, imgSize[0]*imgSize[1]],name='x_data')
            y_ = tf.placeholder(tf.float32, shape=[None, N_CLASS],name='y_data')

        '''*************************************************
        *                                                  *
        *                 create model                     *
        *                                                  *
        *************************************************'''


        # input layer
        x_image = tf.reshape(x, [-1, imgSize[0],imgSize[1], 1])
        tf.summary.image('input', x_image, N_CLASS)
        keep_prob = tf.placeholder(tf.float32)

        if model is 'CNN':
            CNN = TenzorCNN()
            y_pred,WGTH = CNN.CNN2(x_image,CNN_HIDDEN_LAYER,KERNEL_SIZE,POOL_SIZE,STRIDE_SIZE,imgSize)

        elif model is 'LATENT':
            DAE = TenzorAE()
            x_image = tf.reshape(x, [-1, imgSize[0]*imgSize[1]])
            output,latent = DAE.AE(x_image,hidden_layers=AE_HIDDEN_LAYER,keep_prob=keep_prob)
            NN = TenzorNN()
            shape = int(AE_HIDDEN_LAYER[len(AE_HIDDEN_LAYER)//2])
            y_pred = NN.NeuralNetwork(latent,NN_HIDDEN_LAYER,1,keep_prob=keep_prob,shape=shape,fc_neu=N_CLASS)
        else:
            sys.stderr.write('MODEL CONFUSE')
            sys.exit(-1)

        with tf.name_scope('cross_entropy'):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))
                if model is 'LATENT':
                    loss = tf.reduce_mean(tf.squared_difference(x, output))
        with tf.name_scope('gradient_descent_learning_algorithm'):
                if LEARNING_ALGORITHM is 'GRAD':
                    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
                elif LEARNING_ALGORITHM is 'ADAM':
                    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
                if model is 'LATENT':
                    if LEARNING_ALGORITHM is 'GRAD':
                        train_step_loss = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
                    elif LEARNING_ALGORITHM is 'ADAM':
                        train_step_loss = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        with tf.name_scope('evaluation'):
            pred_class = tf.argmax(y_pred, 1)
            correct_prediction = tf.equal(pred_class, tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    '''*************************************************
    *                                                  *
    *                  train model                     *
    *                                                  *
    *************************************************'''

    if 1:#(SAVE_PATH is not None) or (GETT_PATH is not None):
        saver = tf.train.Saver()


    with tf.Session() as sess:
        if TENSOR_BOARD:
            writer = tf.summary.FileWriter("output", sess.graph)
        sess.run(tf.global_variables_initializer())
        if GETT_PATH != None:
            saver.restore(sess, GETT_PATH+model+'.ckpt')
            print("Get model from path: %s" % GETT_PATH+model+'.ckpt')

        if model is 'CNN':
            n_main = 1 # CNN run only one time
        else:
            n_main = 2 # AE run two times for AE 1 and NN 1

        # list that store accuracy
        epoch_acc = []
        if PROGRAM is PROG_PERF:
            if model is 'CNN':
                test_case = randint(0,len(testingSet[1]))
                image = testingSet[0][test_case]
                image = np.array(image)
                image = np.reshape(image,(imgSize[0],imgSize[1]))*255
                cv2.imwrite('originalImage.jpg',image)
                for i in range(0,2):
                    WGHT = WGTH[i].eval(feed_dict={x: [testingSet[0][test_case]], y_: [testingSet[1][test_case]], keep_prob: 1.0})
                    for j in range(0,10):
                        z = np.array(WGHT[0][:,:,j])
                        zmn = z[...].min()
                        zmx = z[...].max()
                        Z = 255*(z-zmn)/(zmx-zmn)
                        cv2.imwrite('D:\\2560\\FRA361_Robot_Studio\\FIBO_project_Module8-9\\layer'+str(i)+'\\weigthImage'+str(j)+'.jpg',Z)
        elif PROGRAM is PROG_TEST:
            cap = cv2.VideoCapture(1)
            while(True):

                # Capture frame-by-frame
                ret, frame = cap.read()
                org = copy.deepcopy(frame)
                # Our operations on the frame come here
                org,LoM = Get_Plate(frame)
                LoM = np.array(LoM)
                # Display the resulting frame
                LoM = np.array(LoM)
                LoC = copy.deepcopy(LoM)
                LoC = LoC//255
                LoC = np.reshape(LoC,(LoC.shape[0],30*60))

                LoC = pred_class.eval(feed_dict={x: LoC, keep_prob: 1.0})

                for i in range(0,len(LoM)):
                    LoMi = cv2.resize(LoM[i],(300,150))
                    cv2.imshow('output_'+str(i)+'_'+str(LoC[i]),LoMi,)
                    cv2.moveWindow('output'+str(i),300*i,80)
                cv2.imshow('original',org)
                if cv2.waitKey(3) & 0xFF == ord('q'):
                    break
                for i in range(0,len(LoM)):
                    cv2.destroyWindow('output_'+str(i)+'_'+str(LoC[i]))



        if TENSOR_BOARD:
            writer.close()

        return accuracy.eval(feed_dict={x: validationSet[0], y_: validationSet[1], keep_prob: 1.0})

if model is 'CNN':
    HL = CNN_HIDDEN_LAYER
elif model is 'LATENT':
    HL = AE_HIDDEN_LAYER



accuracy = main(model = model,aug=AUGMENT,value=AUG_VALUE,GETT_PATH = GETT_PATH,SAVE_PATH=SAVE_PATH,MAIN_HIDDEN_LAYER = HL,NN_HIDDEN_LAYER = NN_HIDDEN_LAYER,
         BATCH_SIZE = BATCH_SIZE,BATCH2PRINT = BATCH2PRINT,EPOCH = EPOCH,LEARNING_RATE = LEARNING_RATE,KEEP_PROB = KEEP_PROB)
print('accuracy of this model is ',accuracy)