import numpy as np
#from skimage import io as io
import poissonblending
from ops import *
import cv2
import tqdm

def poisson_blend(imgs1, imgs2, mask):
        # call this while performing consistency experiment
        out = np.zeros(imgs1.shape)

        #for i in range(0, len(imgs1)):
        img1 = (imgs1 + 1.) / 2.0
        img2 = (imgs2 + 1.) / 2.0
        out = np.clip((poissonblending.blend(img1, img2,  1 - mask) - 0.5) * 2, -1.0, 1.0)

        return out.astype(np.float32)

#x_raw = np.load('x_raw.npy')
#x_inpaint = np.load('x_inpaint.npy')
#x_mask = 1 - np.load('x_mask.npy')

#print("Raw: {}_{} | Inpaint: {}_{} | Mask: {}_{}".format(x_raw.max(), x_raw.min(), x_inpaint.max(), x_inpaint.min($
#nRows = 2
#nCols = 2
#batchSz = 4

#ori_img_name = os.path.join('blending_test', 'ori.png')
#inpaint_img_name = os.path.join('blending_test', 'inpaint.png')
#mask_img_name =  os.path.join('blending_test', 'mask.png')
#blend_img_name = os.path.join('blending_test', 'blend.png')


#x_blend = poisson_blend(x_raw, x_inpaint, x_mask)

#save_images(x_raw[:batchSz,:,:,:], [nRows,nCols], ori_img_name)
#save_images(x_inpaint[:batchSz,:,:,:], [nRows,nCols], inpaint_img_name)
#save_images(x_mask[:batchSz,:,:,:], [nRows,nCols], mask_img_name)
#save_images(x_blend[:batchSz,:,:,:], [nRows,nCols], blend_img_name)

#for i in range(0,4):
#    mask = 255 * np.squeeze(x_mask[i])
#    print("Mask shape:", mask.shape)
#    io.imsave(os.path.join('blending_test', 'mask_{}.png').format(i),mask)

out_loc='/home/avisek/vineel/glcic-master/3_in_1_video/test_output/blend/'

x_raw = np.load('/home/avisek/vineel/glcic-master/3_in_1_video/blending/x_raw.npy')
x_inpaint = np.load('/home/avisek/vineel/glcic-master/3_in_1_video/blending/x_gen.npy')
x_mask = 1 - np.load('/home/avisek/vineel/glcic-master/3_in_1_video/blending/x_mask.npy')

step_num = len(x_raw)

for i in tqdm.tqdm(range(step_num)):

	x_blend = poisson_blend(x_raw[i], x_inpaint[i], x_mask[i])
	x_blend = np.array((x_blend + 1) * 127.5, dtype=np.uint8)
        x_blend = cv2.cvtColor(x_blend, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(out_loc)+'blend'+str(i) + '.jpg', (x_blend))
