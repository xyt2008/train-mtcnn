set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_L_net_onlylandmark.py --lr 0.01 --image_set img_cut_celeba_all --end_epoch 5000 --prefix model/lnet --lr_epoch 2000,4000 --batch_size 100 --thread_num 10 --frequent 100 
pause 