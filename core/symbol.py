import mxnet as mx
import negativemining
import negativemining_landmark
import negativemining_onlylandmark
import negativemining_onlylandmark10
from config import config


def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2,2), num_filter=8, name="conv1", no_bias=True)
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    #cur size: 9x9

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), num_filter=8, num_group=8, name="conv2_dw", no_bias=True)
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep", no_bias=True)
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")
    #cur size: 7x7
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw", no_bias=True)
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep", no_bias=True)
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")
    #cur size: 3x3

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw", no_bias=True)
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    #cur size: 1x1

    conv4_1 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    conv4_2 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=4, name="conv4_2")

    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=conv4_1, mode="channel", name="cls_prob")
        bbox_pred = conv4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = conv4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = conv4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1,1), num_filter=16, name="conv1", no_bias=True)
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=16, num_group=16, name="conv2_dw", no_bias=True)
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep", no_bias=True)
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw", no_bias=True)
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
	
    conv5_1 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=2, name="conv5_1")
    conv5_2 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=4, name="conv5_2")

    cls_prob = mx.symbol.SoftmaxOutput(data=conv5_1, label=label, use_ignore=True,
                                       name="cls_prob")
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxOutput(data=conv5_1, label=label, use_ignore=True, name="cls_prob")
        bbox_pred = conv5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv5_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group

def O_Net(mode="train", with_landmark = False):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    if with_landmark:
        type_label = mx.symbol.Variable(name="type_label")
        landmark_target = mx.symbol.Variable(name="landmark_target")
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),pad=(1,1), num_filter=32, name="conv1")
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw", no_bias=True)
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw", no_bias=True)
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=256, num_group=256, name="conv6_dw", no_bias=True)
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)
        cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)

        conv6_3 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=10, name="conv6_3")	
        bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
        if mode == "test":
            bbox_pred = bn6_2
            landmark_pred = bn6_3
            group = mx.symbol.Group([cls_prob, bbox_pred, landmark_pred])
        else:
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                type_label=type_label, op_type='negativemining_landmark', name="negative_mining")
            group = mx.symbol.Group([out])
    else:
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),pad=(1,1), num_filter=16, name="conv1")
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=16, num_group=16, name="conv2_dw", no_bias=True)
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw", no_bias=True)
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv4_dw", no_bias=True)
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv5_dw", no_bias=True)
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=128, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=128, num_group=128, name="conv6_dw", no_bias=True)
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)
        cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
        if mode == "test":
            bbox_pred = bn6_2
            group = mx.symbol.Group([cls_prob, bbox_pred])
        else:
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                               op_type='negativemining', name="negative_mining")
            group = mx.symbol.Group([out])
    return group

lnet_basenum=32
def L_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    """
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name="conv1") #48/46
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
    conv1_1 = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=32, name="conv1_1") #46/45
    bn1_1 = mx.sym.BatchNorm(data=conv1_1, name='bn1_1', fix_gamma=False,momentum=0.9)
    prelu1_1 = mx.symbol.LeakyReLU(data=bn1_1, act_type="prelu", name="prelu1_1")

    conv1_2 = mx.symbol.Convolution(data=prelu1_1, kernel=(3, 3), stride=(2,2), num_filter=32, name="conv1_2") #45/22
    bn1_2 = mx.sym.BatchNorm(data=conv1_2, name='bn1_2', fix_gamma=False,momentum=0.9)
    prelu1_2 = mx.symbol.LeakyReLU(data=bn1_2, act_type="prelu", name="prelu1_2")

    conv2 = mx.symbol.Convolution(data=prelu1_2, kernel=(3, 3), num_filter=64, name="conv2") #22/20
    bn2 = mx.sym.BatchNorm(data=conv2, name='bn2', fix_gamma=False,momentum=0.9)
    prelu2 = mx.symbol.LeakyReLU(data=bn2, act_type="prelu", name="prelu2")
    
    conv2_1 = mx.symbol.Convolution(data=prelu2, kernel=(2, 2), num_filter=64, name="conv2_1") #20/19
    bn2_1 = mx.sym.BatchNorm(data=conv2_1, name='bn2_1', fix_gamma=False,momentum=0.9)
    prelu2_1 = mx.symbol.LeakyReLU(data=bn2_1, act_type="prelu", name="prelu2_1")

    conv2_2 = mx.symbol.Convolution(data=prelu2_1, kernel=(3, 3), stride=(2,2), num_filter=64, name="conv2_2") #19/9
    bn2_2 = mx.sym.BatchNorm(data=conv2_2, name='bn2_2', fix_gamma=False,momentum=0.9)
    prelu2_2 = mx.symbol.LeakyReLU(data=bn2_2, act_type="prelu", name="prelu2_2")
	
    conv3 = mx.symbol.Convolution(data=prelu2_2, kernel=(3, 3), num_filter=64, name="conv3") #9/7
    bn3 = mx.sym.BatchNorm(data=conv3, name='bn3', fix_gamma=False,momentum=0.9)
    prelu3 = mx.symbol.LeakyReLU(data=bn3, act_type="prelu", name="prelu3")
    
    conv3_1 = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=64, name="conv3_1") #7/5
    bn3_1 = mx.sym.BatchNorm(data=conv3_1, name='bn3_1', fix_gamma=False,momentum=0.9)
    prelu3_1 = mx.symbol.LeakyReLU(data=bn3_1, act_type="prelu", name="prelu3_1")
    
    conv4 = mx.symbol.Convolution(data=prelu3_1, kernel=(3, 3), num_filter=128, name="conv4") # 5/3
    bn4 = mx.sym.BatchNorm(data=conv4, name='bn4', fix_gamma=False,momentum=0.9)
    prelu4 = mx.symbol.LeakyReLU(data=bn4, act_type="prelu", name="prelu4")

    conv5 = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=256, name="conv5") # 3/1
    bn5 = mx.sym.BatchNorm(data=conv5, name='bn5', fix_gamma=False,momentum=0.9)
    prelu5 = mx.symbol.LeakyReLU(data=bn5, act_type="prelu", name="prelu5")
	"""
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet_basenum, name="conv1") 
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="avg", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1")

    conv2_dw = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv2_dw")
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv2_sep")
    prelu2_sep = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2_sep")
    pool2 = mx.symbol.Pooling(data=prelu2_sep, pool_type="avg", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool2")

    conv3_dw = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv3_dw")
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv3_sep")
    prelu3_sep = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3_sep")
    pool3 = mx.symbol.Pooling(data=prelu3_sep, pool_type="avg", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool3")

    conv4_dw = mx.symbol.Convolution(data=pool3, kernel=(2, 2), num_filter=lnet_basenum*4,num_group=lnet_basenum*4, name="conv4_dw")
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv4_sep")
    prelu4_sep = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv5_dw")
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv5_sep")
    prelu5_sep = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5_sep")
	
    conv6_3 = mx.symbol.FullyConnected(data=prelu5_sep, num_hidden=10, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        if config.use_landmark10:
            target_x1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=0, end=1)
            target_x2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=1, end=2)
            target_x3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=2, end=3)
            target_x4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=3, end=4)
            target_x5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=4, end=5)
            target_y1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=5, end=6)
            target_y2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=6, end=7)
            target_y3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=7, end=8)
            target_y4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=8, end=9)
            target_y5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=9, end=10)
            bn_x1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=0, end=1)
            bn_x2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=1, end=2)
            bn_x3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=2, end=3)
            bn_x4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=3, end=4)
            bn_x5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=4, end=5)
            bn_y1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=5, end=6)
            bn_y2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=6, end=7)
            bn_y3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=7, end=8)
            bn_y4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=8, end=9)
            bn_y5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=9, end=10)
            pred_x1 = mx.symbol.LinearRegressionOutput(data=bn_x1, label=target_x1, grad_scale=1, name="pred_x1")			
            pred_x2 = mx.symbol.LinearRegressionOutput(data=bn_x2, label=target_x2, grad_scale=1, name="pred_x2")			
            pred_x3 = mx.symbol.LinearRegressionOutput(data=bn_x3, label=target_x3, grad_scale=1, name="pred_x3")			
            pred_x4 = mx.symbol.LinearRegressionOutput(data=bn_x4, label=target_x4, grad_scale=1, name="pred_x4")			
            pred_x5 = mx.symbol.LinearRegressionOutput(data=bn_x5, label=target_x5, grad_scale=1, name="pred_x5")			
            pred_y1 = mx.symbol.LinearRegressionOutput(data=bn_y1, label=target_y1, grad_scale=1, name="pred_y1")			
            pred_y2 = mx.symbol.LinearRegressionOutput(data=bn_y2, label=target_y2, grad_scale=1, name="pred_y2")			
            pred_y3 = mx.symbol.LinearRegressionOutput(data=bn_y3, label=target_y3, grad_scale=1, name="pred_y3")			
            pred_y4 = mx.symbol.LinearRegressionOutput(data=bn_y4, label=target_y4, grad_scale=1, name="pred_y4")			
            pred_y5 = mx.symbol.LinearRegressionOutput(data=bn_y5, label=target_y5, grad_scale=1, name="pred_y5")			
            out = mx.symbol.Custom(pred_x1=pred_x1,pred_x2=pred_x2,pred_x3=pred_x3,pred_x4=pred_x4,pred_x5=pred_x5,
                                pred_y1=pred_y1,pred_y2=pred_y2,pred_y3=pred_y3,pred_y4=pred_y4,pred_y5=pred_y5,
                                target_x1=target_x1,target_x2=target_x2,target_x3=target_x3,target_x4=target_x4,target_x5=target_x5,
                                target_y1=target_y1,target_y2=target_y2,target_y3=target_y3,target_y4=target_y4,target_y5=target_y5,
                                op_type='negativemining_onlylandmark10', name="negative_mining") 
            group = mx.symbol.Group([out])
        else:
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                op_type='negativemining_onlylandmark', name="negative_mining")
            group = mx.symbol.Group([out])
        
    return group