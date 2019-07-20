import glob
import os
import cv2
import numpy as np
from skimage.transform import AffineTransform,warp

ratio = 0.98
image_size = 128


x = []
paths = glob.glob('/home/avisek/vineel/data/celebA/img_align_celeba/*')
cnt = 0
cnt1 = 1
for path in paths:
    cnt += 1
    img = cv2.imread(path)
    if (cnt%2==0):
        t = 1
    else:
        t = -1
    for z in range(4,-5,-1):
        if (z==0):
            img = cv2.resize(img, (image_size, image_size))
            #cv2.imwrite('/home/avisek/vineel/data/'+'original_%05d.jpg' % (cnt*9+cnt2), (img))
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x.append(img1)
            continue
        transform = AffineTransform(translation=(z*t,0))
        img0 = warp(img, transform, mode='wrap', preserve_range=True)
        img0 = img0.astype(img.dtype)
        img0 = cv2.resize(img0, (image_size, image_size))
        #cv2.imwrite('/home/avisek/vineel/data/'+'original_%05d.jpg' % (cnt*9+cnt2), (img0))
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        x.append(img0)
    if cnt>8000:
        cnt = 0
        x = np.array(x, dtype=np.uint8)
        np.save('/home/avisek/vineel/glcic-master/9_in_7_video/npy_celebA/x_train_'+str(cnt1)+'.npy', x)
        print(len(x))
        x = []
        cnt1 += 1
x = np.array(x, dtype=np.uint8)
np.save('/home/avisek/vineel/glcic-master/9_in_7_video/npy_celebA/x_test.npy', x)
print(len(x))
