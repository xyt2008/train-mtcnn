import mxnet as mx
import numpy as np
from config import config


class LANDMARK_MSE(mx.metric.EvalMetric):
    def __init__(self):
        super(LANDMARK_MSE, self).__init__('lmL2')

    def update(self,labels, preds):
        # output: pred_x1, keep_x1,pred_x2, keep_x2,pred_x3, keep_x3,pred_x4, keep_x4,pred_x5, keep_x5,
        #         pred_y1, keep_y1,pred_y2, keep_y2,pred_y3, keep_y3,pred_y4, keep_y4,pred_y5, keep_y5
        # label: landmark_target
        landmark_target = labels[0].asnumpy()
        for i in range(10):
            pred_delta = preds[i*2].asnumpy()
            landmark_keep = preds[i*2+1].asnumpy()
            keep = np.where(landmark_keep == 1)[0]

            pred_delta = pred_delta[keep].reshape(-1,1)
            target = landmark_target[keep][:,i].reshape(-1,1)

            e = (pred_delta - target)**2
            error = np.sum(e)
            self.sum_metric += error
            self.num_inst += e.size

class LANDMARK_L1(mx.metric.EvalMetric):
    def __init__(self):
        super(LANDMARK_L1, self).__init__('lmL1')

    def update(self,labels, preds):
        # output: pred_x1, keep_x1,pred_x2, keep_x2,pred_x3, keep_x3,pred_x4, keep_x4,pred_x5, keep_x5,
        #         pred_y1, keep_y1,pred_y2, keep_y2,pred_y3, keep_y3,pred_y4, keep_y4,pred_y5, keep_y5
        # label: landmark_target
        #print labels
        #print preds
        landmark_target = labels[0].asnumpy()
        for i in range(10):
            pred_delta = preds[i*2].asnumpy()
            landmark_keep = preds[i*2+1].asnumpy()
            keep = np.where(landmark_keep == 1)[0]

            pred_delta = pred_delta[keep].reshape(-1,1)
            target = landmark_target[keep][:,i].reshape(-1,1)
            #print 'pred_delta='
            #print pred_delta
            #print 'target='
            #print target
            e = abs(pred_delta - target)
            error = np.sum(e)
            #print 'e='
            #print e
            #print error, e.size
            self.sum_metric += error
            self.num_inst += e.size