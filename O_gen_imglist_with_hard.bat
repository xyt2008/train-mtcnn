python prepare_data\gen_imglist20_with_hard.py --size 48
copy prepare_data\rnet\train_48_with_hard.txt data\mtcnn\imglists
del data\cache\mtcnn_train_48_with_hard_gt_roidb.pkl
pause