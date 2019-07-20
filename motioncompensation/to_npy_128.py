import glob
import os
import cv2
import numpy as np

ratio = 0.98
image_size = 128

tfrecord_ind = 1

x_train = []

if not os.path.exists('/home/avisek/vineel/glcic-master/motion_compensation/npy_128'):
    os.mkdir('/home/avisek/vineel/glcic-master/motion_compensation/npy_128')

paths = glob.glob('/home/avisek/vineel/data/train/*')
for path in paths:
    word = path.split('\\')[-1]
    path1 = glob.glob(word+'/video/*')
    print('processing  '+word+'...')
    for path2 in path1:
        word1 = path2.split('\\')[-1]
        path3 = glob.glob(word1+'/*')
        for path4 in sorted(path3):
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(word2)
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_train.append(img)
	x_train = np.array(x_train, dtype=np.uint8)
	np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_128/x_train_128_'+str(tfrecord_ind)+'.npy', x_train)
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
	np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_128/x_train_mc_128_'+str(tfrecord_ind)+'.npy', x_train_mc)

	tfrecord_ind += 1
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
        for path4 in sorted(path3):
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(word2)
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_test.append(img)
	x_test = np.array(x_test, dtype=np.uint8)
        np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_128/x_test_128_'+str(tfrecord_ind)+'.npy', x_test)

	size_test = len(x_test)
        x_test_mc = []
        for i in range(5, size_test-5):
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
        np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_128/x_test_mc_128_'+str(tfrecord_ind)+'.npy', x_test_mc)

        tfrecord_ind += 1
        x_test = []
