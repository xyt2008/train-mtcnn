set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_O_net.py --with_landmark --lr 0.005 --image_set train_48_with_hard_landmark_1 --end_epoch 30 --prefix model/onet --lr_epoch 8,14,100 --batch_size 2000 --thread_num 32 --frequent 100 
pause 