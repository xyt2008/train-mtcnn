import numpy as np
from easydict import EasyDict as edict

config = edict()
config.root = 'C:/mtcnn'

config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = True
config.BBOX_OHEM_RATIO = 0.7
config.LANDMARK_OHEM = False
config.LANDMARK_OHEM_RATIO = 0.7
config.landmark_L1_thresh = 0.02
config.landmark_L1_outlier_thresh = 1

config.EPS = 1e-14
config.enable_gray = True
config.use_landmark10 = True
