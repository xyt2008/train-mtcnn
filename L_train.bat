set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_L_net.py --lr 0.001 --image_set landmark --end_epoch 16 --prefix model/lnet --lr_epoch 8,14,100 --batch_size 2000 --thread_num 32 --frequent 100 
pause 