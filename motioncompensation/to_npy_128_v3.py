import os
import numpy as np

dir_='/home/avisek/vineel/glcic-master/motion_compensation/npy_hmdb/train/npy'

#x_train = np.load(os.path.join(dir_, 'x_train_128_p1.npy'))
#print(x_train.shape)
#size_train = len(x_train)
#print(x_train[5,5,5,1])
#x_train = np.array([a / 127.5 - 1 for a in x_train])
#print(x_train[5,5,5,1,1])
#print(x_train[10].shape)
#x_train1 = x_train[0:size_train/2]
#x_train2 = x_train[size_train/2:size_train]
#print(x_train2.shape)
#print(x_train1.shape)
#size_train1 = len(x_train1)
#np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_128/x_train_mc_128_p1.npy', x_train)
for k in range(1,6512):
	x_train = np.load(os.path.join(dir_, 'x_train_128_'+str(k)+'.npy'))
	print(x_train.shape)
	size_train = len(x_train)
	print(x_train[5,5,5,1])
        x_train_mc = []
	#x_train_mc = np.array(x_train_mc)
        for i in range(1, size_train-1):
                l = []
                #ind = np.random.randint(-5,-1)
                l.append(x_train[i-1])
                l.append(x_train[i])
                #ind = np.random.randint(1,5)
                l.append(x_train[i+1])
                l = np.array(l)
                #print(l.shape)
                l = np.moveaxis(l, 0, -1)
                #print(l.shape)
                x_train_mc.append(l)
        x_train_mc = np.array(x_train_mc)
        print(x_train_mc.shape)
	print(x_train_mc[5,5,5,1,1])
	x_train_mc = np.array([a / 127.5 - 1 for a in x_train_mc])
	print(x_train_mc[5,5,5,1,1])
	np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_hmdb/train/npy_mc/x_train_mc_128_'+str(k)+'.npy', x_train_mc)

#x_train = np.array([a / 127.5 - 1 for a in x_train]
#for i in range(len(x_train)):
#	x_train[i] = x_train[i]/127.5 -1
#print(x_train.shape)
#leng = len(x_train)
#x_train1 = x_train[0:leng/2]
#x_train2 = x_train[leng/2:leng]
#print(x_train2.shape)
#print(x_train1.shape)
#np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_128/x_train_mc_128_p1.npy', x_train1)
#np.save('/home/avisek/vineel/glcic-master/motion_compensation/npy_128/x_train_mc_128_p2.npy', x_train2)
