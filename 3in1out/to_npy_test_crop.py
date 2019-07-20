import glob
import os
import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils

image_size = 128
detector = dlib.get_frontal_face_detector()
cnt = 0
paths = glob.glob('/home/avisek/vineel/glcic-master/3_in_1_video/test_data/*')
for path in paths:
    head, subject = os.path.split(path)
    path1 = glob.glob(path+'/video/*')
    print('processing  '+path+'...')
    x_test = []
    cnt += 1
    for path2 in path1:
        head, pos = os.path.split(path2)
        path3 = glob.glob(path2+'/*')
        print(path2)
        if pos in ['head','head2','head3']:
            continue
        cnt1 = 0
        for path4 in sorted(path3):
            cnt1 += 1
            if cnt1 > 80:
                break
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(word2)
            rects = detector(img,1)
            x,y,w,h = np.array([0,0,512,384])
            for (k, rect) in enumerate(rects):
                x,y,w,h = np.array([rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()])
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
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_test.append(img)
    x_test = np.array(x_test, dtype=np.uint8)
    np.save('/home/avisek/vineel/glcic-master/5_in_3_video/npy_128_crop/x_test_128_'+str(cnt)+'.npy', x_test)
    print(x_test.shape)

