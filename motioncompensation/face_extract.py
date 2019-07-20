import glob
import os
import cv2
import numpy as np
from imutils import face_utils
import imutils
import dlib


detector = dlib.get_frontal_face_detector()

ratio = 0.98
image_size = 128

tfrecord_ind = 0

x_train = []

if not os.path.exists('/home/avisek/vineel/glcic-master/motion_compensation/npy_128'):
    os.mkdir('/home/avisek/vineel/glcic-master/motion_compensation/npy_128')

out_loc = '/home/avisek/vineel/glcic-master/motion_compensation/face/'

paths = glob.glob('/home/avisek/vineel/data/train/*')
for path in paths:
    word = path.split('\\')[-1]
    path1 = glob.glob(word+'/video/*')
    print('processing  '+word+'...')
    for path2 in path1:
        word1 = path2.split('\\')[-1]
        path3 = glob.glob(word1+'/*')
	cnt = 0
	tfrecord_ind += 1
	print(tfrecord_ind)
	if (os.path.exists('/home/avisek/vineel/glcic-master/motion_compensation/npy_face/x_train_128_'+str(tfrecord_ind)+'.npy')):
		continue
        for path4 in sorted(path3):
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(word2)
	    print(img.shape)
            rects = detector(img,1)
            x,y,w,h = np.array([0,0,512,384])
            for (k, rect) in enumerate(rects):
                x,y,w,h = np.array([rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()])
		print(x)
		print(y)
		print(w)
		print(h)
	    if (x <= 0 or y<=0 or h==0 or w==0):
	    	x = 100
		y = 30
		h = 260
		w = 325
	    else:
            	if (x - 30> 0):
                	x = x - 30
            	if (y > 50):
                	y = y - int(3*y/4)
            	if (w+60 < 512):
                	w = w+60
            	if (h+int(3*y/4) < 384):
                	h = h+int(3*y/4)+30
            img = img[y:y+h,x:x+w,:]
	    print(x)
            print(y)
            print(w)
            print(h)
            img = cv2.resize(img, (image_size, image_size))
	    #print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_train.append(img)
	    cnt += 1
            #cv2.imwrite(str(out_loc)+str(cnt)+'.jpg' ,(img))
	x_train = np.array(x_train, dtype=np.uint8)
        np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_face/x_train_128_'+str(tfrecord_ind)+'.npy', x_train)
        print(x_train.shape)

	size_train = len(x_train)
        x_train_mc = []
        for i in range(5, size_train-5):
                l = []
                ind = np.random.randint(-5,-1)
                l.append(x_train[i+ind])
                l.append(x_train[i])
                ind = np.random.randint(1,5)
                l.append(x_train[i+ind])
                l = np.array(l)
                #print(l.shape)
                l = np.moveaxis(l, 0, -1)
                #print(l.shape)
                x_train_mc.append(l)
        x_train_mc = np.array(x_train_mc)
        print(x_train_mc.shape)
        np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_face/x_train_mc_128_'+str(tfrecord_ind)+'.npy', x_train_mc)

        #tfrecord_ind += 1
        x_train = []

tfrecord_ind = 1
x_test = []

paths = glob.glob('/home/avisek/vineel/data/test/*')

for path in paths:
    word = path.split('\\')[-1]
    path1 = glob.glob(word+'/video/*')
    print('processing  '+word+'...')
    for path2 in path1:
        word1 = path2.split('\\')[-1]
        path3 = glob.glob(word1+'/*')
        cnt = 0
        for path4 in sorted(path3):
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(word2)
            print(img.shape)
            rects = detector(img,1)
            x,y,w,h = np.array([0,0,512,384])
            for (k, rect) in enumerate(rects):
                x,y,w,h = np.array([rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()])
            if (x - 30> 0):
                x = x - 30
            if (y > 50):
                y = y - int(3*y/4)
            if (w+60 < 512):
                w = w+60
            if (h+int(3*y/4) < 384):
                h = h+int(3*y/4)+30
            img = img[y:y+h,x:x+w,:]
            img = cv2.resize(img, (image_size, image_size))
            #print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_test.append(img)
            #cv2.imwrite(str(out_loc)+str(cnt)+'.jpg' ,(img))
        x_test = np.array(x_test, dtype=np.uint8)
        np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_face/x_test_128_'+str(tfrecord_ind)+'.npy', x_test)
        print(x_test.shape)

        size_train = len(x_test)
        x_test_mc = []
        for i in range(5, size_train-5):
                l = []
                ind = np.random.randint(-5,-1)
                l.append(x_test[i+ind])
                l.append(x_test[i])
                ind = np.random.randint(1,5)
                l.append(x_test[i+ind])
                l = np.array(l)
                #print(l.shape)
                l = np.moveaxis(l, 0, -1)
                #print(l.shape)
                x_test_mc.append(l)
        x_test_mc = np.array(x_test_mc)
        print(x_test_mc.shape)
        np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_face/x_test_mc_128_'+str(tfrecord_ind)+'.npy', x_test_mc)

        tfrecord_ind += 1
        x_test = []

