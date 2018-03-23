import tensorflow as tf
import cv2
from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from Tenzor import TenzorCNN,TenzorNN,TenzorAE
import os
img = IP.font_to_image("arial.ttf", 32, index=0, string="two")
k = IP.get_plate(img, (60, 30))
k = k[0]
img = IP.binarize(k.UnrotateWord, IP.SAUVOLA_THRESHOLDING, 19)

# cv2.imshow("plate", img)
#
# cv2.waitKey(0)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# parallel operation
CPU_NUM = 0

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
GETT_PATH = "\\CNN_MODEL_A\\"
SAVE_PATH = "\\CNN_MODEL_A\\"


BATCH2PRINT = 20
EPOCH = 2000
AUGMENT = AUG_NONE
DIVIDEY = 1
VALIDATE_SECTION = 100





LEARNING_ALGORITHM = 'ADAM'
imgSize = [30, 60]
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
HL = CNN_HIDDEN_LAYER

def main(model='CNN',aug=0,value=None,GETT_PATH = None,SAVE_PATH=None,MAIN_HIDDEN_LAYER = [],NN_HIDDEN_LAYER = [],
         BATCH_SIZE = 16,BATCH2PRINT = 1000,EPOCH = 1,LEARNING_RATE = 0.01,KEEP_PROB = 0.9):


    global CNN_MODEL,LATENT_MODEL,LEARNING_ALGORITHM
    fin_AE = False
    if model is 'CNN':
        pass
    elif model is 'LATENT':
        pass
    # else:
    #     sys.stderr.write('MODEL ERROR: '+ str(model))
    #     sys.exit(-1)

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

        # elif model is 'LATENT':
        #     DAE = TenzorAE()
        #     x_image = tf.reshape(x, [-1, imgSize[0]*imgSize[1]])
        #     output,latent = DAE.AE(x_image,hidden_layers=AE_HIDDEN_LAYER,keep_prob=keep_prob)
        #     NN = TenzorNN()
        #     shape = int(AE_HIDDEN_LAYER[len(AE_HIDDEN_LAYER)//2])
        #     y_pred = NN.NeuralNetwork(latent,NN_HIDDEN_LAYER,1,keep_prob=keep_prob,shape=shape,fc_neu=N_CLASS)
        # else:
        #     sys.stderr.write('MODEL CONFUSE')
        #     sys.exit(-1)

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
        # if TENSOR_BOARD:
        #     writer = tf.summary.FileWriter("output", sess.graph)
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
                LoM = cv2.imread('file',0)
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



        # if TENSOR_BOARD:
        #     writer.close()

        return accuracy.eval(feed_dict={x: validationSet[0], y_: validationSet[1], keep_prob: 1.0})


accuracy = main(model ='CNN',aug=AUGMENT,value=AUG_VALUE,GETT_PATH = GETT_PATH,SAVE_PATH=SAVE_PATH,MAIN_HIDDEN_LAYER = HL,NN_HIDDEN_LAYER = NN_HIDDEN_LAYER,
         BATCH_SIZE = BATCH_SIZE,BATCH2PRINT = BATCH2PRINT,EPOCH = EPOCH,LEARNING_RATE = LEARNING_RATE,KEEP_PROB = KEEP_PROB)

# init_op = tf.global_variables_initializer()
# graph=tf.Graph()
# with graph.as_default():
#     saver = tf.train.Saver()
# with tf.Session() as sess:
#
#     saver.restore(sess,"CNN_MODEL_A\\modelCNN.ckpt")
#     n_main=1
    # PROGRAM = PROG_TEST
    #
    # # dataset from 'mnist' or 'project'
    # DATA = 'PROJECT'
    #
    # # choose machine learning model 'LATENT' or 'CNN'
    # model = 'CNN'
