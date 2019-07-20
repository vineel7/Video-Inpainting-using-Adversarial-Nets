import numpy as np
import glob
import os
import tqdm
import dlib

IMAGE_SIZE = 128
IMAGE_DEPTH = 5  # no of frames
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 32
BATCH_SIZE = 8
BATCH_FRAME_SIZE = BATCH_SIZE * IMAGE_DEPTH

detector = dlib.get_frontal_face_detector()
def get_points2(x_batch):
    points = []
    mask = []
    global x,y,w,h
    x,y,w,h=np.array([50,50,60,60])
    for i in range(BATCH_FRAME_SIZE):
        image = np.array((x_batch[i] + 1) * 127.5, dtype=np.uint8)
        if i>0:
             points.append([x1, y1, x2, y2])
             m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
             m[int(q1):int(q2) + 1, int(p1):int(p2) + 1] = 1
             mask.append(m)
             continue
        rects = detector(image,1)
        for (k, rect) in enumerate(rects):
            x,y,w,h = np.array([rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()])
        Face_Hole_max_w = min(HOLE_MAX,int(w))
        Face_Hole_max_h = min(HOLE_MAX,int(h))
        L = np.random.randint(HOLE_MIN, Face_Hole_max_w)
        M = np.random.randint(HOLE_MIN, Face_Hole_max_h)
        p1 = x + np.random.randint(0, w - L)
        q1 = y + np.random.randint(0, h - M)
        p2 = p1 + L
        q2 = q1 + M
        p3 = p1 + int(L/2)
        q3 = q1 + int(M/2)
        x1 = p3 - LOCAL_SIZE/2
        if(x1<1):
            t = 1 - x1;
            x1 = 1;
            p2 = p2 + t;
            p1 = p1 + t;
            p3 = p3 + t;
        y1 = q3 - LOCAL_SIZE/2
        if(y1<1):
            t = 1 - y1;
            y1 = 1;
            q2 = q2 + t;
            q1 = q1 + t;
            q3 = q3 + t;
        x2 = p3 + LOCAL_SIZE/2
        if(x2>127):
            t = x2 -127;
            x2 = 127;
            p2 = p2 - t;
            p1 = p1 - t;
            x1 = x1 - t;
        y2 = q3 + LOCAL_SIZE/2
        if(y2>127):
            t = y2 -127;
            y2 = 127;
            q2 = q2 - t;
            q1 = q1 - t;
            y1 = y1 - t;
        points.append([x1, y1, x2, y2])
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[int(q1):int(q2) + 1, int(p1):int(p2) + 1] = 1
        mask.append(m)


    return np.array(points), np.array(mask)

if __name__ == '__main__':

    for c in range(8):
        x_test = np.load('/home/avisek/vineel/glcic-master/5_in_3_video/npy_128_crop/x_test_128_'+str(c+1)+'.npy')
        x_test = np.array([a / 127.5 - 1 for a in x_test])
        step_num = 12
        mask = []
        points = []
        offset = 0
        for i in tqdm.tqdm(range(step_num)):
            x_batch = []
            for j in range(BATCH_SIZE):
                for k in range(5):
                    x_batch.append(x_test[offset+k])
                offset = offset+3
            x_batch = np.array(x_batch)
            points_batch, mask_batch = get_points2(x_batch)
            mask.append(mask_batch)
            points.append(points_batch)
        mask = np.array(mask)
        points = np.array(points)
        print(mask.shape)
        print(points.shape)
        np.save('/home/avisek/vineel/glcic-master/5_in_3_video/npy_mask/x_mask_'+str(c+1)+'.npy',mask)
        np.save('/home/avisek/vineel/glcic-master/5_in_3_video/npy_mask/x_points_'+str(c+1)+'.npy',points)
