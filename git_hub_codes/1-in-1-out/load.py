import os
import numpy as np

def load(dir_='/home/vineel/video_inpainting_glcic/glcic-master/baseline_codes/npy/'):
    x_train = np.load(os.path.join(dir_, 'x_train.npy'))
    x_test = np.load(os.path.join(dir_, 'x_test.npy'))
    return x_train, x_test


if __name__ == '__main__':
    x_train, x_test = load()
    print(x_train.shape)
    print(x_test.shape)

