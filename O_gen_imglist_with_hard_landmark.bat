python prepare_data\gen_imglist.py --size 48 --with_hard --with_landmark
copy prepare_data\48\train_48_with_hard_landmark.txt data\mtcnn\imglists
copy prepare_data\onet\landmark_48.txt data\mtcnn\imglists
pause