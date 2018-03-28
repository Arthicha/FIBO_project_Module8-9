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
import mpl_toolkits.mplot3d.axes3d as p3

# system module
import os
import sys
import cv2
import copy

# my own library
from Tenzor import TenzorCNN,TenzorNN,TenzorAE


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

# dataset from 'mnist' or 'project'
DATA = 'PROJECT'

# choose machine learning model 'LATENT' or 'CNN'
model = 'CNN'

# continue previous model
CONTINUE = False

# save and get path
GETT_PATH = None#"D:\\2560\\FRA361_Robot_Studio\\FIBO_project_Module8-9\\model"
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
CONV_PIC = False
PLOT_LATENT = False




'''*************************************************
*                                                  *
*                hyper parameter                   *
*                                                  *
*************************************************'''

if DATA is 'MNIST':
    imgSize = [28,28] # size of image
    N_CLASS = 10
elif DATA is 'PROJECT':
    imgSize = [64,128]
    N_CLASS = 30


CNN_HIDDEN_LAYER = [36,64,128,256,512,512] #amount of layer > 3
NN_HIDDEN_LAYER = [1,1]
AE_HIDDEN_LAYER = [imgSize[0]*imgSize[1],100,50,3,50,100,imgSize[0]*imgSize[1]]
KERNEL_SIZE = [[7,7],[7,7],[7,7],[5,5],[5,5]]
POOL_SIZE = [[4,4],[2,2],[2,2],None,[2,2]]
STRIDE_SIZE = [4,2,2,0,2]

BATCH_SIZE = 2000

LEARNING_RATE = 0.001
KEEP_PROB = 0.9
MULTI_COLUMN = 1




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
    for s in range(1,3):
        print('STATUS: process data',str(100.0*s/3.0))
        for j in range(0,N_CLASS):

            object = listOfClass[j]
            f = open('data0-9compress\\dataset_'+str(object)+'_'+suffix[s]+'.txt','r')
            image = str(f.read()).split('\n')[:-1]
            f.close()
            delList = []
            for i in range(len(image)):
                image[i] = np.fromstring(image[i], dtype=float, sep=',')
                image[i] = np.array(image[i])
                image[i] = np.reshape(image[i],(imgSize[1]*imgSize[0]))
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

    #a,b = shuffly(TestTrainValidate[0],LabelTTT[0])
    #testingSet = [np.array(a),np.array(b)]

    #a,b = shuffly(TestTrainValidate[2],LabelTTT[2])
    #validationSet = [np.array(a),np.array(b)]
    testingSet  = [TestTrainValidate[2],LabelTTT[2]]
    validationSet = [TestTrainValidate[2],LabelTTT[2]]
print('training size:',len(trainingSet[1]))
'''*************************************************
*                                                  *
*                     function                     *
*                                                  *
*************************************************'''

# plot graph of relationship between accuracy and epoch then save it
def accuracyPlot(y,graphName,xLabel,yLabel,saveAs):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(y)
    plt.title(graphName)
    plt.ylabel(xLabel)
    plt.xlabel(yLabel)
    ax.legend()
    fig.savefig(saveAs+'.png')

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

# list of image augmentation
def augm(LoD,mod=0,valu=None):

    LoND = []
    LoD = np.reshape(LoD,(-1,imgSize[0],imgSize[1]))
    if (valu is None) and (mod is not AUG_NONE):
        sys.stderr.write('AUGMENTATION VALUE ERROR')
        sys.exit(-1)
    LoDc = copy.deepcopy(LoD)
    for img in LoD:
        if mod is not AUG_NONE:
            if mod is AUG_DTSX:
                LoND.append(distorse(img,function='sine',axis='x',alpha=valu[0],beta=valu[1]))
            elif mod is AUG_DTSY:
                LoND.append(distorse(img,function='sine',axis='y',alpha=valu[0],beta=valu[1]))
            elif mod is AUG_DTSB:
                img2 = distorse(img,function='sine',axis='x',alpha=valu[0],beta=valu[1])
                LoND.append(distorse(img2,function='sine',axis='y',alpha=valu[0],beta=valu[1]))
        elif mod is AUG_NONE:
            LoND = LoD
    if SHOW_AUG:
        for i in range(0,len(LoND)):
            cv2.imshow('org',LoDc[i])
            cv2.imshow('img',LoND[i])
            cv2.waitKey(0)
    LoND = np.reshape(LoND,(-1,imgSize[0]*imgSize[1]))
    return LoND

'''*************************************************
*                                                  *
*                 input section                    *
*                                                  *
*************************************************'''
def main(best_accuracy,model='CNN',aug=0,value=None,GETT_PATH = None,SAVE_PATH=None,MAIN_HIDDEN_LAYER = [],NN_HIDDEN_LAYER = [],
         BATCH_SIZE = 16,BATCH2PRINT = 1000,EPOCH = 1,LEARNING_RATE = 0.01,KEEP_PROB = 0.9):

    global CNN_MODEL,LATENT_MODEL,LEARNING_ALGORITHM,imgSize
    fin_AE = False
    if model is 'CNN':
        CNN_MODEL = 1
        LATENT_MODEL = 0
    elif model is 'LATENT':
        CNN_MODEL = 0
        LATENT_MODEL = 1
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

        if (CNN_MODEL) and (not LATENT_MODEL):
            CNN = TenzorCNN()
            y_pred,WGTH = CNN.CNN2(x_image,CNN_HIDDEN_LAYER,KERNEL_SIZE,POOL_SIZE,STRIDE_SIZE,imgSize)
            '''pool = CNN.CNN(x_image,CNN_HIDDEN_LAYER,keep_prob=keep_prob,pool=True,stride=2)
            shape = pool.shape[1]
            shape = int(shape*shape*MULTI_COLUMN)
            NN = TenzorNN()
            y_pred = NN.NeuralNetwork(pool,NN_HIDDEN_LAYER,CNN_HIDDEN_LAYER[-1],keep_prob=keep_prob,shape=shape,fc_neu=N_CLASS)
            '''
        elif (not CNN_MODEL) and (LATENT_MODEL):
            DAE = TenzorAE()
            x_ae = tf.reshape(x, [-1, imgSize[0]*imgSize[1]])
            y_pred,latent = DAE.AE(x_ae,hidden_layers=AE_HIDDEN_LAYER,keep_prob=keep_prob)
        else:
            sys.stderr.write('MODEL CONFUSE')
            sys.exit(-1)
        with tf.name_scope('cross_entropy'):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))
                if (not CNN_MODEL) and (LATENT_MODEL):
                    loss = tf.reduce_mean(tf.squared_difference(x, y_pred))
        with tf.name_scope('gradient_descent_learning_algorithm'):

                if (not CNN_MODEL) and (LATENT_MODEL):
                    if LEARNING_ALGORITHM is 'GRAD':
                        train_step_loss = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
                    elif LEARNING_ALGORITHM is 'ADAM':
                        train_step_loss = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
                else:
                    if LEARNING_ALGORITHM is 'GRAD':
                        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
                    elif LEARNING_ALGORITHM is 'ADAM':
                        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        if (CNN_MODEL) and (not LATENT_MODEL):
            with tf.name_scope('evaluation'):
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            accuracy = loss
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
            if (CNN_MODEL) and (not LATENT_MODEL):
                filename = 'CNN'
            elif fin_AE:
                filename = 'AE'
            saver.restore(sess, GETT_PATH+filename+'.ckpt')
            print("Get model from path: %s" % GETT_PATH+filename+'.ckpt')

        input('>>>>')
        #len(mnist.train.labels)

        # list that store accuracy
        epoch_acc = []

        for epoch in range(EPOCH):
            if 1:
                for i in range(BATCH_SIZE,len(trainingSet[1])//DIVIDEY,BATCH_SIZE):

                    batch = [trainingSet[0][i-BATCH_SIZE:i],trainingSet[1][i-BATCH_SIZE:i]]

                    # do augmentation
                    batch[0] = augm(batch[0],mod=aug,valu=value)

                    if (i//BATCH_SIZE) % BATCH2PRINT == 0:
                        if ((CNN_MODEL) and (not LATENT_MODEL)):
                            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                            validation_accuracy = 0
                            n_sec = 0
                            for k in range(0,len(validationSet[1]),len(validationSet[1])//VALIDATE_SECTION):
                                validation_accuracy += accuracy.eval(feed_dict={x: validationSet[0][k:k+len(validationSet[1])//VALIDATE_SECTION], y_: validationSet[1][k:k+len(validationSet[1])//VALIDATE_SECTION], keep_prob: 1.0})
                                n_sec += 1
                            validation_accuracy = validation_accuracy/n_sec
                        else:
                            train_accuracy = 1.00-loss.eval(feed_dict={x: batch[0],keep_prob: 1.0})
                            validation_accuracy = 0
                            n_sec = 0
                            for k in range(0,len(validationSet[1]),len(validationSet[1])//VALIDATE_SECTION):
                                validation_accuracy += 1.00-loss.eval(feed_dict={x: validationSet[0][k:k+len(validationSet[1])//VALIDATE_SECTION],keep_prob: 1.0})
                                n_sec += 1
                            validation_accuracy = validation_accuracy/n_sec

                        print('EPOCH %d: step %d, training accuracy %g, validation accuracy %g' % (epoch,i, train_accuracy,validation_accuracy))
                    if ((CNN_MODEL) and (not LATENT_MODEL)):
                        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB})
                    else:
                        train_step_loss.run(feed_dict={x: batch[0], keep_prob: KEEP_PROB})


                if ((CNN_MODEL) and (not LATENT_MODEL)):
                    testing_accuracy = 0
                    n_sec = 0
                    for i in range(0,len(testingSet[1]),len(testingSet[1])//VALIDATE_SECTION):
                        testing_accuracy += accuracy.eval(feed_dict={x: testingSet[0][i:i+len(testingSet[1])//VALIDATE_SECTION], y_: testingSet[1][i:i+len(testingSet[1])//VALIDATE_SECTION], keep_prob: 1.0})
                        n_sec += 1
                    testing_accuracy = testing_accuracy/n_sec
                else:
                    testing_accuracy = 0
                    n_sec = 0
                    for i in range(0,len(testingSet[1]),len(testingSet[1])//VALIDATE_SECTION):
                        testing_accuracy += 1.00-loss.eval(feed_dict={x: testingSet[0][i:i+len(testingSet[1])//VALIDATE_SECTION],keep_prob: 1.0})
                        n_sec += 1
                    testing_accuracy = testing_accuracy/n_sec
                print('EPOCH %d: test accuracy %g' % (epoch,testing_accuracy))

                if 1:
                    validation_accuracy = 0
                    n_sec = 0
                    for i in range(0,len(validationSet[1]),len(validationSet[1])//VALIDATE_SECTION):
                        validation_accuracy += accuracy.eval(feed_dict={x: validationSet[0][i:i+len(validationSet[1])//VALIDATE_SECTION], y_: validationSet[1][i:i+len(validationSet[1])//VALIDATE_SECTION], keep_prob: 1.0})
                        n_sec += 1
                    validation_accuracy = validation_accuracy/n_sec
                    print('model accuracy:', validation_accuracy)
                    if (SAVE_PATH != None): #and (validation_accuracy >= best_accuracy):
                        save = True
                        best_accuracy = validation_accuracy
                        if (CNN_MODEL) and (not LATENT_MODEL):
                            filename = 'CNN'
                        else:
                            filename = 'AE'
                        if save:
                            epoch_acc.append(float(validation_accuracy))
                            accuracyPlot(epoch_acc,'accuracy of the model in each epoch','accuracy (prob)','time (epoch)','accuracy'+str(filename))
                            f = open('model_'+filename+'_config.txt','w')
                            f.write(str(float(validation_accuracy))+'\n')
                            if filename == 'AE':
                                for each_layer in AE_HIDDEN_LAYER:
                                    f.write(str(each_layer)+',')
                                f.write('\n')
                            elif filename == 'CNN':
                                for each_layer in CNN_HIDDEN_LAYER:
                                    f.write(str(each_layer)+',')
                                f.write('\n')
                            for each_layer in NN_HIDDEN_LAYER:
                                f.write(str(each_layer)+',')
                            f.write('\n')
                            f.write(str(BATCH_SIZE)+'\n')
                            f.write(str(KEEP_PROB)+'\n')
                            f.write(str(LEARNING_RATE)+'\n')
                            f.close()
                            save_path = saver.save(sess, SAVE_PATH+str(filename)+'.ckpt')
                            print("Model saved in path: %s" % save_path)

                            if PLOT_LATENT:
                                if ((not CNN_MODEL) and (LATENT_MODEL)):
                                    fig = plt.figure()
                                    ls = p3.Axes3D(fig)

                                    def get_cmap(n, name='hsv'):
                                        return plt.cm.get_cmap(name, n)

                                    cmap = get_cmap(30)
                                    for i in range(0,len(trainingSet[1])//100):
                                        latent_point = latent.eval(feed_dict={x: [trainingSet[0][i]],keep_prob: 1.0})
                                        clas = trainingSet[1][i]
                                        clas = np.argmax(np.array(clas))
                                        ls.scatter(xs=[latent_point[0][0]],ys=[latent_point[0][1]],zs=[latent_point[0][2]],c=cmap(clas))
                                    plt.title('')
                                    plt.ylabel('dimension 1')
                                    plt.xlabel('dimension 2')
                                    ls.legend()
                                    fig.savefig('D:\\2560\\FRA361_Robot_Studio\\FIBO_project_Module8-9\\latentSpace.png')



                            if CONV_PIC:
                                if ((CNN_MODEL) and (not LATENT_MODEL)):

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
                                            cv2.imwrite('\layer'+str(i)+'\weigthImage'+str(j)+'.jpg',Z)

        if TENSOR_BOARD:
            writer.close()
        return accuracy.eval(feed_dict={x: validationSet[0], y_: validationSet[1], keep_prob: 1.0})

def rand(min,max):
    return ran.random()*(max-min)+min

def randint(min,max):
    return ran.randint(min,max)
# hyper parameter



# best_accuracy is for comparing with previous models
best_accuracy = 0.00
print('STATUS: main program have started')
try:
    if model is 'CNN':
        filename = 'CNN'
    elif model is 'LATENT':
        filename = 'AE'

    if CONTINUE:
        # get configuration of previous model
        f = open('model_'+filename+'_config.txt','r')
        read = f.read()
        f.close()
        read = read.split('\n')
        best_accuracy = float(read[0])
    else:
        best_accuracy = 0.00
except:
    best_accuracy = 0.00
if(1):
    print('STATUS: model connfiguration')
    print('STATUS: droupout',KEEP_PROB)
    print('STATUS: learning rate',LEARNING_RATE)
    print('STATUS: batch',BATCH_SIZE)
    if model is 'CNN':
        HL = CNN_HIDDEN_LAYER
        print('STATUS: cnn_layer',CNN_HIDDEN_LAYER)
    elif model is 'LATENT':
        HL = AE_HIDDEN_LAYER
        print('STATUS: ae_layer',AE_HIDDEN_LAYER)
    # sart main program
    accuracy = main(best_accuracy, model = model,aug=AUGMENT,value=AUG_VALUE,GETT_PATH = GETT_PATH,SAVE_PATH=SAVE_PATH,MAIN_HIDDEN_LAYER = HL,NN_HIDDEN_LAYER = NN_HIDDEN_LAYER,
         BATCH_SIZE = BATCH_SIZE,BATCH2PRINT = BATCH2PRINT,EPOCH = EPOCH,LEARNING_RATE = LEARNING_RATE,KEEP_PROB = KEEP_PROB)


    if accuracy > best_accuracy:
        best_accuracy = accuracy



