import cv2
import numpy as np
import os
import glob

image_size = 128

def extractFrames(pathIn, pathOut,record_ind,np_path):

    #os.mkdir(pathOut)

    cap = cv2.VideoCapture(pathIn)
    count = 0
    x_train = []

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            #print('Read %d frame: ' % count, ret)
            #cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            count += 1
	    img = cv2.resize(frame, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_train.append(img)
        else:
            break
    x_train = np.array(x_train, dtype=np.uint8)
    np.save(np_path+'x_train_128_'+str(record_ind)+'.npy', x_train)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

	train_count = 0
	test_count = 0
	train_path = '/home/avisek/vineel/data/hmdb/*'
	train_frame = '/home/avisek/vineel/data/hmdb/train_frame/'
	np_train = '/home/avisek/vineel/glcic-master/motion_compensation/npy_hmdb/train/'
	np_test = '/home/avisek/vineel/glcic-master/motion_compensation/npy_hmdb/test/'
	paths = glob.glob(train_path)
	for path in paths:
		path1 = glob.glob(path+'/*')
		print(path)
		count = 0
		for vid in path1:
			count += 1
			if (count <= 5):
				test_count += 1
				extractFrames(vid, path,test_count,np_test)
			else:
				train_count += 1
				extractFrames(vid, path,train_count,np_train)
