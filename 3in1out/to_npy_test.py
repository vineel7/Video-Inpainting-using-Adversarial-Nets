import glob
import os
import cv2
import numpy as np

image_size = 128

cnt = 0
paths = glob.glob('/home/avisek/vineel/glcic-master/3_in_1_video/test_data/*')
for path in sorted(paths):
    cnt += 1
    x_train = []
    word = path.split('\\')[-1]
    path1 = glob.glob(word+'/video/*')
    print('processing  '+word+'...')
    for path2 in sorted(path1):
        word1 = path2.split('\\')[-1]
        path3 = glob.glob(word1+'/*')
	cnt1 = 0
        for path4 in sorted(path3):
	    cnt1 += 1
	    if cnt1 > 60:
		break
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(path4)
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_train.append(img)
    x_train = np.array(x_train, dtype=np.uint8)
    print(len(x_train))
    if not os.path.exists('/home/avisek/vineel/glcic-master/3_in_1_video/npy_test'):
    	os.mkdir('/home/avisek/vineel/glcic-master/3_in_1_video/npy_test')
    np.save('/home/avisek/vineel/glcic-master/3_in_1_video/npy_test/x_test'+str(cnt)+'.npy', x_train)

