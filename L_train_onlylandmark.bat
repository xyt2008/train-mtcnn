set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_L_net_onlylandmark.py --lr 0.0001 --image_set good --end_epoch 10000 --prefix model/lnet --lr_epoch 5000,8000 --batch_size 100 --thread_num 10 --frequent 100 
pause 