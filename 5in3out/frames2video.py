import cv2
import numpy as np
import os
import glob

from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    #files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    #files.sort(key = lambda x: int(x[5:-4]))
    paths = glob.glob(pathIn)
    for filename in sorted(paths):
        #filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
def main():
    pathIn= '/home/avisek/vineel/glcic-master/5_in_3_video/thesis_results/test_output_2m/p5/masked/*'
    pathOut = '/home/avisek/vineel/glcic-master/5_in_3_video/thesis_results/test_output_2m/maskvideo.avi'
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)
if __name__=="__main__":
    main()
