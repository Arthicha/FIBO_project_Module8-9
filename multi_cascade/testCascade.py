# ~/virtualenv/ROBOTICS_studios/bin/python



'''*************************************************
*                                                  *
*             import class & library               *
*                                                  *
*************************************************'''
import sys
import os

import cv2
from detectCascade import multiCascade

def connectCamera():
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()

        if ret == None:
            sys.exit("can't connect camera")
        else :
            frame2 = hc.detectFromCascade(frame)
            cv2.imshow('frame',frame)
            cv2.imshow('frame2',frame2)
            if cv2.waitKey(1)& 0xff == ord('q'):
                cv2.destroyAllWindows()
                cap.release()

                break

def main():

    inputKey = sys.argv[1:3]
    hc = multiCascade()
    # connectCamera()

    if inputKey == [] or str(inputKey[0]) == 'help' :
        sys.exit('test_run.py [param1] [param2]\nparam1:\t 0 or removeAllCascade \n\t 1 or renewCascade\n\t 2 or testCascade\nparam2:\t HAAR / HOG / LBP\n')

    elif str(inputKey[0]) == '0' or str(inputKey[0]) == 'removeAllCascade' :
        hc.deleteCascadeFile()

    elif str(inputKey[0]) == '1' or str(inputKey[0]) == 'renewCascade' :
        '''remove old cascade files and copy new cascade files.'''

        print('remove old cascade files and copy new cascade files.')
        hc.deleteCascadeFile(feature= [str(inputKey[1])])
        hc.copyCascadeFile(feature= str(inputKey[1]))
        print('test')
        hc.testCascade(feature= str(inputKey[1]))

    elif str(inputKey[0]) == '2' or str(inputKey[0]) == 'testCascade':
        '''test cascade accuracy file files.'''

        print('test cascade accuracy files.')
        hc.testCascade(feature= str(inputKey[1]))

    # elif str(inputKey[0]) == '3' or str(inputKey[0]) == 'removeAllCascade' :
    #     '''remove all main cascade files.'''

    #     print('remove all main cascade files.')
    #     hc.deleteMainCascadeFile()

if __name__ == '__main__':
    main()