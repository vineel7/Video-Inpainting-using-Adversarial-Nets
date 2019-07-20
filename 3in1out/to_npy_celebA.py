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
    transform = AffineTransform(translation=(5,0))
    img0 = warp(img, transform, mode='wrap', preserve_range=True)
    img0 = img0.astype(img.dtype)
    transform1 = AffineTransform(translation=(-5,0))
    img1 = warp(img, transform1, mode='wrap', preserve_range=True)
    img1 = img1.astype(img.dtype)
    img0 = cv2.resize(img0, (image_size, image_size))
    img1 = cv2.resize(img1, (image_size, image_size))
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    x.append(img0)
    x.append(img)
    x.append(img1)
    if cnt>25000:
        cnt = 0
        x = np.array(x, dtype=np.uint8)
        np.save('/home/avisek/vineel/glcic-master/3_in_1_video/npy_celebA/x_train_'+str(cnt1)+'.npy', x)
        print(len(x))
        x = []
        cnt1 += 1
x = np.array(x, dtype=np.uint8)
np.save('/home/avisek/vineel/glcic-master/3_in_1_video/npy_celebA/x_test.npy', x)
print(len(x))
