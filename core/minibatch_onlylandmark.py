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
    annotation = annotation_line.strip().split(' ')
    img_path = config.root+'/data/%s/'%config.landmark_img_set+annotation[0]
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    bbox = np.array(annotation[1:5],dtype=np.float32)
    landmark = np.array(annotation[5:15],dtype=np.float32)
    dis1 = (landmark[0] - landmark[8])*(landmark[0] - landmark[8])+(landmark[1] - landmark[9])*(landmark[1] - landmark[9])
    dis2 = (landmark[2] - landmark[6])*(landmark[2] - landmark[6])+(landmark[3] - landmark[7])*(landmark[3] - landmark[7])
    dis = max(dis1,dis2)
    dis = dis**0.5
    x1, y1, w, h = bbox
    cx = landmark[4]
    cy = landmark[5]
    bbox_size = int(0.25*(abs(x1-cx)+abs(x1+w-cx)+abs(y1-cy)+abs(y1+h-cy)))
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
        rot_landmark = image_processing.rotateLandmark(landmark, cur_angle,1)
        cur_size = int(npr.randint(5, 25)*0.1*bbox_size)
        border_size = int(cur_size*0.05)

        # delta here is the offset of box center
        delta_x = npr.randint(-int(w * 0.2), int(w * 0.2)+1)
        delta_y = npr.randint(-int(h * 0.2), int(h * 0.2)+1)

        nx1 = int(max(x1 + w / 2 + delta_x - cur_size / 2, 0))
        ny1 = int(max(y1 + h / 2 + delta_y - cur_size / 2, 0))
        nx2 = nx1 + cur_size
        ny2 = ny1 + cur_size

        if nx2 > width or ny2 > height:
            continue
        ignore = 0
        max_x_landmark = -1
        min_x_landmark = width+1
        max_y_landmark = -1
        min_y_landmark = height+1
        for j in range(5):
            if rot_landmark[j*2] < nx1+border_size or rot_landmark[j*2] >= nx1 + cur_size-border_size:
                ignore = 1
            if rot_landmark[j*2+1] < ny1+border_size or rot_landmark[j*2+1] >= ny1 + cur_size-border_size:
                ignore = 1
            if max_x_landmark < rot_landmark[j*2]:
                max_x_landmark = rot_landmark[j*2]
            if min_x_landmark > rot_landmark[j*2]:
                min_x_landmark = rot_landmark[j*2]
            if max_y_landmark < rot_landmark[j*2+1]:
                max_y_landmark = rot_landmark[j*2+1]
            if min_y_landmark > rot_landmark[j*2+1]:
                min_y_landmark = rot_landmark[j*2+1]
												
        if ignore == 1:
            continue
        landmark_x_dis = max_x_landmark - min_x_landmark
        landmark_y_dis = max_y_landmark - min_y_landmark
        tmp_dis = landmark_x_dis*landmark_x_dis + landmark_y_dis*landmark_y_dis
        if tmp_dis < 0.04*cur_size*cur_size:
            continue
        offset_x1 = (rot_landmark[0] - nx1 + 0.5) / float(cur_size)
        offset_y1 = (rot_landmark[1] - ny1 + 0.5) / float(cur_size)
        offset_x2 = (rot_landmark[2] - nx1 + 0.5) / float(cur_size)
        offset_y2 = (rot_landmark[3] - ny1 + 0.5) / float(cur_size)
        offset_x3 = (rot_landmark[4] - nx1 + 0.5) / float(cur_size)
        offset_y3 = (rot_landmark[5] - ny1 + 0.5) / float(cur_size)
        offset_x4 = (rot_landmark[6] - nx1 + 0.5) / float(cur_size)
        offset_y4 = (rot_landmark[7] - ny1 + 0.5) / float(cur_size)
        offset_x5 = (rot_landmark[8] - nx1 + 0.5) / float(cur_size)
        offset_y5 = (rot_landmark[9] - ny1 + 0.5) / float(cur_size)

        rot_img,_ = image_processing.rotateWithLandmark(img,landmark, cur_angle,1)
        cropped_im = rot_img[ny1 : ny2, nx1 : nx2, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
        
        cur_sample_num += 1

    if force_accept == 1:
        ny1 = max(0,y1)
        ny2 = min(height,y1+h)
        nx1 = max(0,x1)
        nx2 = min(width,x1+w)
        w = nx2-nx1
        h = ny2-ny1
        cropped_im = img[ny1 : ny2, nx1 : nx2, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
        offset_x1 = (rot_landmark[0] - x1 + 0.5) / float(w)
        offset_y1 = (rot_landmark[1] - y1 + 0.5) / float(h)
        offset_x2 = (rot_landmark[2] - x1 + 0.5) / float(w)
        offset_y2 = (rot_landmark[3] - y1 + 0.5) / float(h)
        offset_x3 = (rot_landmark[4] - x1 + 0.5) / float(w)
        offset_y3 = (rot_landmark[5] - y1 + 0.5) / float(h)
        offset_x4 = (rot_landmark[6] - x1 + 0.5) / float(w)
        offset_y4 = (rot_landmark[7] - y1 + 0.5) / float(h)
        offset_x5 = (rot_landmark[8] - x1 + 0.5) / float(w)
        offset_y5 = (rot_landmark[9] - y1 + 0.5) / float(h)
    
    landmark = [offset_x1,offset_x2,offset_x3,offset_x4,offset_x5,offset_y1,offset_y2,offset_y3,offset_y4,offset_y5]

    if npr.randint(0,2) == 1:
        landmark[0], landmark[1] = 1.0-landmark[1], 1.0-landmark[0]
        landmark[2] = 1.0-landmark[2]
        landmark[3], landmark[4] = 1.0-landmark[4], 1.0-landmark[3]
        landmark[5], landmark[6] = landmark[6], landmark[5]
        landmark[8], landmark[9] = landmark[9], landmark[8]
        resized_im = resized_im[:, ::-1, :]
		
    if config.enable_blur:
        kernel_size = npr.randint(-5,5)*2+1
        if kernel_size >= 3:
            blur_im = cv2.GaussianBlur(resized_im,(kernel_size,kernel_size),0)
            resized_im = blur_im

    return resized_im,landmark