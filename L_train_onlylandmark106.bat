set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_L_net_onlylandmark106.py --lr 0.01 --image_set landmark106 --end_epoch 5000 --lr_epoch 1000,1500 --batch_size 100 --thread_num 10 --frequent 10 
pause 