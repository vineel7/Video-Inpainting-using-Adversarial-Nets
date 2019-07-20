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

x_train = []

if not os.path.exists('/home/avisek/vineel/glcic-master/5_in_3_video/npy_128_crop'):
    os.mkdir('/home/avisek/vineel/glcic-master/5_in_3_video/npy_128_crop')

outloc = '/home/avisek/vineel/data/face_crop/train/'
paths = glob.glob('/home/avisek/vineel/data/train/*')
cnt = 1
for path in paths:
    head, subject = os.path.split(path)
    path1 = glob.glob(path+'/video/*')
    #print(subject)
    print('processing  '+path+' ...')
    out_loc1 = '/home/avisek/vineel/data/face_crop/train/'+str(subject)+'/'
    if not os.path.exists(out_loc1):
        os.mkdir(out_loc1)
    for path2 in path1:
        head, pos = os.path.split(path2)
        path3 = glob.glob(path2+'/*')
        #print(pos)
        print(path2)
        if pos in ['head','head2','head3']:
            continue
        out_loc2 = out_loc1 + str(pos)+'/'
        if not os.path.exists(out_loc2):
            os.mkdir(out_loc2)
        cnt2 = 1
        for path4 in sorted(path3):
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(word2)
            #print(img.shape)
            rects = detector(img,1)
            x,y,w,h = np.array([0,0,512,384])
            for (k, rect) in enumerate(rects):
                x,y,w,h = np.array([rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()])
                #print(x)
                #print(y)
                #print(w)
                #print(h)
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
            #print(x)
            #print(y)
            #print(w)
            #print(h)
            img = cv2.resize(img, (image_size, image_size))
            #print(img.shape)
            cv2.imwrite(str(out_loc2)+'%05d.jpg' % cnt2,(img))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_train.append(img)
            cnt2 += 1
x_train = np.array(x_train, dtype=np.uint8)
np.save('/home/avisek/vineel/glcic-master/5_in_3_video/npy_128_crop/x_train_128.npy', x_train)
print(x_train.shape)

x_test = []
paths = glob.glob('/home/avisek/vineel/data/test/*')
for path in paths:
    head, subject = os.path.split(path)
    path1 = glob.glob(path+'/video/*')
    print('processing  '+path+'...')
    out_loc1 = '/home/avisek/vineel/data/face_crop/test/'+str(subject)+'/'
    if not os.path.exists(out_loc1):
        os.mkdir(out_loc1)
    for path2 in path1:
        head, pos = os.path.split(path2)
        path3 = glob.glob(path2+'/*')
        print(path2)
        if pos in ['head','head2','head3']:
            continue
        out_loc2 = out_loc1 + str(pos)+'/'
        if not os.path.exists(out_loc2):
            os.mkdir(out_loc2)
        cnt2 = 1
        for path4 in sorted(path3):
            word2 = path4.split('\\')[-1]
            img = cv2.imread(word2)
            print(word2)
            print(img.shape)
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
            #print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_test.append(img)
            cv2.imwrite(str(out_loc2)+'%05d.jpg' % cnt2,(img))
            cnt2 += 1
x_test = np.array(x_test, dtype=np.uint8)
np.save('/home/avisek/vineel/glcic-master/5_in_3_video/npy_128_crop/x_test_128.npy', x_test)
print(x_test.shape)
