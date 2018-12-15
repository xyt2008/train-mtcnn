import cv2
import numpy as np

img_path = 'img_celeba'
out_img_path = 'img_cut_celeba'

with open('list_bbox_celeba.txt', 'r') as f:
    bbox_lines = f.readlines()

with open('list_landmarks_celeba.txt', 'r') as f:
    landmark_lines = f.readlines()
f = open('img_cut_celeba_all.txt','w')
num = len(bbox_lines)-2
for i in range(num):
    bbox_line = bbox_lines[i+2].split('\n')[0]
    landmark_line = landmark_lines[i+2].split('\n')[0]
    bbox_splits = bbox_line.split()
    landmark_splits = landmark_line.split()
    img = cv2.imread(img_path+'/'+bbox_splits[0])
    width = img.shape[1]
    height = img.shape[0]
    bbox = np.array(bbox_splits[1:5],dtype=np.float32)
    landmark = np.array(landmark_splits[1:11],dtype=np.float32)
    cx,cy = landmark[4:6]
    x1,y1,w,h = bbox
    bbox_ori = bbox.copy()
    bbox_size = int(0.25*(abs(x1-cx)+abs(x1+w-cx)+abs(y1-cy)+abs(y1+h-cy)))
    x1 = int(cx - bbox_size*0.5)
    y1 = int(cy - bbox_size*0.5)
    w = bbox_size
    h = bbox_size
    off_x = int(max(0,x1-1.5*w))
    off_y = int(max(0,y1-1.5*h))
    max_x = int(min(width,x1+w*2.5))
    max_y = int(min(height,y1+h*2.5))
    bbox[0] = max(0,x1)
    bbox[1] = max(0,y1)	
    bbox[2] = x1+w - bbox[0]
    bbox[3] = y1+h - bbox[1]
    bbox_ori[0] -= off_x
    bbox_ori[1] -= off_y
    for j in range(5):
        landmark[j*2] -= off_x
        landmark[j*2+1] -= off_y
    if bbox[2] >= 20 and bbox[3] >= 20 and bbox[0]+bbox[2] <= width and bbox[1]+bbox[3] <= height:
        cut_width = max_x-off_x
        cut_height = max_y-off_y
        cut_img = img[off_y:max_y,off_x:max_x,:]
        if cut_width > 200 and cut_height > 200:
            scale = 200.0/min(cut_width,cut_height)
            dst_width = int(cut_width*scale)
            dst_height = int(cut_height*scale)
            cut_img = cv2.resize(cut_img,(dst_width,dst_height),interpolation=cv2.INTER_LINEAR)
            bbox_ori *= scale
            landmark *= scale
        cv2.imwrite(out_img_path+'/'+bbox_splits[0],cut_img)
        line = '%s %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f \n'%(bbox_splits[0],bbox_ori[0],bbox_ori[1],bbox_ori[2],bbox_ori[3],
               landmark[0],landmark[1],landmark[2],landmark[3],landmark[4],
               landmark[5],landmark[6],landmark[7],landmark[8],landmark[9])
        f.write(line)
    if (i+1)%100 == 0:
        print i+1
	
f.close()