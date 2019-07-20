import glob
import os
import cv2
import numpy as np

ratio = 0.98
image_size = 128

x_train = []
paths = glob.glob('/home/vineel/video_inpainting_glcic/data/train/*')
for path in paths:
    word = path.split('\\')[-1]
    path1 = glob.glob(word+'/video/*')
    print('processing  '+word+'...')
    for path2 in path1:
        word1 = path2.split('\\')[-1]
        path3 = glob.glob(word1+'/*')
        for path4 in path3:
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(word2)
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_train.append(img)
x_train = np.array(x_train, dtype=np.uint8)
np.random.shuffle(x_train)
x_test = []
paths = glob.glob('/home/vineel/video_inpainting_glcic/data/test/*')
for path in paths:
    word = path.split('\\')[-1]
    path1 = glob.glob(word+'/video/*')
    print('processing  '+word+'...')
    for path2 in path1:
        word1 = path2.split('\\')[-1]
        path3 = glob.glob(word1+'/*')
        for path4 in path3:
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(word2)
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_test.append(img)
x_test = np.array(x_test, dtype=np.uint8)
np.random.shuffle(x_test)
print(len(x_train))
print(len(x_test))
if not os.path.exists('/home/vineel/video_inpainting_glcic/glcic-master/baseline_codes/npy'):
    os.mkdir('/home/vineel/video_inpainting_glcic/glcic-master/baseline_codes/npy')
np.save('/home/vineel/video_inpainting_glcic/glcic-master/baseline_codes/npy/x_train.npy', x_train)
np.save('/home/vineel/video_inpainting_glcic/glcic-master/baseline_codes/npy/x_test.npy', x_test)
