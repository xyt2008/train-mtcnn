import cv2
import threading
from tools import image_processing
import numpy as np
import numpy.random as npr
import math
import os,sys
sys.path.append(os.getcwd())
from config import config
import tools.image_processing as image_processing

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.ims, self.landmarks = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.ims, self.landmarks
        except Exception:
            return None

def get_minibatch_thread(imdb, im_size):
    num_images = len(imdb)
    processed_ims = list()
    landmark_reg_target = list()
    #print(num_images)
    for i in range(num_images):
        im,landmark = augment_for_one_image(imdb[i],im_size)
        im_tensor = image_processing.transform(im,True)
        processed_ims.append(im_tensor)
        landmark_reg_target.append(landmark)

    return processed_ims, landmark_reg_target

def get_minibatch(imdb, im_size, thread_num = 4):
    num_images = len(imdb)
    thread_num = max(2,thread_num)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    #print(num_per_thread)
    threads = []
    for t in range(thread_num):
        start_idx = int(num_per_thread*t)
        end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_imdb = [imdb[i] for i in range(start_idx, end_idx)]
        cur_thread = MyThread(get_minibatch_thread,(cur_imdb,im_size))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    processed_ims = list()
    landmark_reg_target = list()

    for t in range(thread_num):
        cur_process_ims, cur_landmark_reg_target = threads[t].get_result()
        processed_ims = processed_ims + cur_process_ims
        landmark_reg_target = landmark_reg_target + cur_landmark_reg_target    
    
    im_array = np.vstack(processed_ims)
    landmark_target_array = np.vstack(landmark_reg_target)
    
    data = {'data': im_array}
    label = {}
    label['landmark_target'] = landmark_target_array

    return data, label

def augment_for_one_image(annotation_line, size):
    annotation = annotation_line.strip().split()
    img_path = config.root+'/data/%s/'%config.landmark_img_set+annotation[0]
    #print img_path
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    landmark = np.array(annotation[1:213],dtype=np.float32)
    landmark_x = landmark[0::2]
    landmark_y = landmark[1::2]
    max_x = max(landmark_x)
    min_x = min(landmark_x)
    max_y = max(landmark_y)
    min_y = min(landmark_y)
    cx = 0.5*(max_x+min_x)
    cy = 0.5*(max_y+min_y)
    h = max_x-min_x
    w = max_y-min_y
    bbox_size = max(h,w)
    x1 = int(cx - bbox_size*0.5)
    y1 = int(cy - bbox_size*0.5)
    w = bbox_size
    h = bbox_size
 

    cur_angle = npr.randint(int(config.min_rot_angle),int(config.max_rot_angle)+1)
    try_num = 0
    cur_sample_num = 0
    base_num = 1
    force_accept = 0
    while cur_sample_num < base_num:
        try_num += 1
        if try_num > base_num*1000:
            force_accept = 1
            break
        rot_landmark_x,rot_landmark_y = image_processing.rotateLandmark106(cx,cy,landmark_x,landmark_y, cur_angle,1)
        cur_size = int(npr.randint(10, 21)*0.1*bbox_size)
        up_border_size = int(-cur_size*0.15)
        down_border_size = int(-cur_size*0.15)
        left_border_size = int(-cur_size*0.15)
        right_border_size = int(-cur_size*0.15)

        # delta here is the offset of box center
        delta_x = npr.randint(-int(w * 0.35), int(w * 0.35)+1)
        delta_y = npr.randint(-int(h * 0.35), int(h * 0.35)+1)

        nx1 = int(max(x1 + w / 2 + delta_x - cur_size / 2, 0))
        ny1 = int(max(y1 + h / 2 + delta_y - cur_size / 2, 0))
        nx2 = nx1 + cur_size
        ny2 = ny1 + cur_size

        if nx2 > width or ny2 > height:
            continue
        ignore = 0
        max_rot_landmark_x = max(rot_landmark_x)
        min_rot_landmark_x = min(rot_landmark_x)
        max_rot_landmark_y = max(rot_landmark_y)
        min_rot_landmark_y = min(rot_landmark_y)
        if min_rot_landmark_x < nx1+left_border_size or max_rot_landmark_x >= nx1 + cur_size-right_border_size:
            ignore = 1
        if min_rot_landmark_y < ny1+up_border_size or max_rot_landmark_y >= ny1 + cur_size-down_border_size:
            ignore = 1
												
        if ignore == 1:
            continue
        landmark_x_dis = max_rot_landmark_x - min_rot_landmark_x
        landmark_y_dis = max_rot_landmark_y - min_rot_landmark_y
        tmp_dis = landmark_x_dis*landmark_x_dis + landmark_y_dis*landmark_y_dis
        if tmp_dis < 0.64*cur_size*cur_size:
            continue
			
        offset_x = (rot_landmark_x - nx1+0.5)/float(cur_size)
        offset_y = (rot_landmark_y - ny1+0.5)/float(cur_size)
        
        rot_img,_,_ = image_processing.rotateWithLandmark106(img,cx,cy,landmark_x,landmark_y, cur_angle,1)
        cropped_im = rot_img[ny1 : ny2, nx1 : nx2, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
        
        cur_sample_num += 1

    if force_accept == 1:
        ny1 = max(0,y1)
        ny2 = int(min(height,y1+h))
        nx1 = max(0,x1)
        nx2 = int(min(width,x1+w))
        w = nx2-nx1
        h = ny2-ny1
        #print ny1,ny2,nx1,nx2
        cropped_im = img[ny1 : ny2, nx1 : nx2, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
        offset_x = (landmark_x - nx1+0.5)/float(cur_size)
        offset_y = (landmark_y - ny1+0.5)/float(cur_size)
    
    if config.landmark106_select23:
        select_landmark=[0 for i in range(46)]
        select_landmark[0:16:2] = offset_x[66:74]
        select_landmark[1:17:2] = offset_y[66:74]
        select_landmark[16] = offset_x[104]
        select_landmark[17] = offset_y[104]
        select_landmark[18:34:2] = offset_x[75:83]
        select_landmark[19:35:2] = offset_y[75:83]
        select_landmark[34] = offset_x[105]
        select_landmark[35] = offset_y[105]
        select_landmark[36] = offset_x[60]
        select_landmark[37] = offset_y[60]
        select_landmark[38] = offset_x[84]
        select_landmark[39] = offset_y[84]
        select_landmark[40] = offset_x[96]
        select_landmark[41] = offset_y[96]
        select_landmark[42] = offset_x[90]
        select_landmark[43] = offset_y[90]
        select_landmark[44] = offset_x[100]
        select_landmark[45] = offset_y[100]
        landmark = select_landmark
    else:
        landmark[0::2] = offset_x
        landmark[1::2] = offset_y
    
    
    if config.enable_blur:
        kernel_size = npr.randint(-5,4)*2+1
        if kernel_size >= 3:
            blur_im = cv2.GaussianBlur(resized_im,(kernel_size,kernel_size),0)
            resized_im = blur_im
    
    return resized_im,landmark