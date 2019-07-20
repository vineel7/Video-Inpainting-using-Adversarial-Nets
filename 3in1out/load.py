import os
import numpy as np

def load(dir_='/home/avisek/vineel/glcic-master/5_in_3_video/npy_128_crop'):
    x_train = np.load(os.path.join(dir_, 'x_train_128.npy'))
    x_test = np.load(os.path.join(dir_, 'x_test_128.npy'))
    return x_train,x_test


if __name__ == '__main__':
    x_train, x_test = load()
    print(x_train.shape)
    print(x_test.shape)

