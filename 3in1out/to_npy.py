import glob
import os
import cv2
import numpy as np

ratio = 0.98
image_size = 256

x_train = []
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
print(len(x_train))
print(len(x_test))
if not os.path.exists('/home/avisek/vineel/glcic-master/3_in_1_video/npy'):
    os.mkdir('/home/avisek/vineel/glcic-master/3_in_1_video/npy')
np.save('/home/avisek/vineel/glcic-master/3_in_1_video/npy/x_train_256.npy', x_train)
np.save('/home/avisek/vineel/glcic-master/3_in_1_video/npy/x_test_256.npy', x_test)
