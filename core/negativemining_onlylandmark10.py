import mxnet as mx
import numpy as np
from config import config

class NegativeMiningOperator_OnlyLandmark10(mx.operator.CustomOp):
    def __init__(self, landmark_ohem=config.LANDMARK_OHEM, landmark_ohem_ratio=config.LANDMARK_OHEM_RATIO):
        super(NegativeMiningOperator_OnlyLandmark10, self).__init__()
        self.landmark_ohem = landmark_ohem
        self.landmark_ohem_ratio = landmark_ohem_ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        #print in_data
        pred_x1 = in_data[0].asnumpy() # batchsize x 1
        pred_x2 = in_data[1].asnumpy() # batchsize x 1
        pred_x3 = in_data[2].asnumpy() # batchsize x 1
        pred_x4 = in_data[3].asnumpy() # batchsize x 1
        pred_x5 = in_data[4].asnumpy() # batchsize x 1
        pred_y1 = in_data[5].asnumpy() # batchsize x 1
        pred_y2 = in_data[6].asnumpy() # batchsize x 1
        pred_y3 = in_data[7].asnumpy() # batchsize x 1
        pred_y4 = in_data[8].asnumpy() # batchsize x 1
        pred_y5 = in_data[9].asnumpy() # batchsize x 1
        target_x1 = in_data[10].asnumpy() # batchsize x 1
        target_x2 = in_data[11].asnumpy() # batchsize x 1
        target_x3 = in_data[12].asnumpy() # batchsize x 1
        target_x4 = in_data[13].asnumpy() # batchsize x 1
        target_x5 = in_data[14].asnumpy() # batchsize x 1
        target_y1 = in_data[15].asnumpy() # batchsize x 1
        target_y2 = in_data[16].asnumpy() # batchsize x 1
        target_y3 = in_data[17].asnumpy() # batchsize x 1
        target_y4 = in_data[18].asnumpy() # batchsize x 1
        target_y5 = in_data[19].asnumpy() # batchsize x 1
		
        # x1,y1
        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[10], req[10], in_data[5])
        keep_x1 = np.zeros(pred_x1.shape[0])
        num = keep_x1.shape[0]
        #print keep_x1,num
        if self.landmark_ohem:
            keep_num = int(num * self.landmark_ohem_ratio)
            L1_error_x = abs(pred_x1 - target_x1)
            L1_error_y = abs(pred_y1 - target_y1)
            L1_error = L1_error_x+L1_error_y
            keep = np.argsort(L1_error.flat)[::-1][:keep_num]
            keep_x1[keep] = 1
            for i in range(num):
                if L1_error[i] < config.landmark_L1_thresh*2 or L1_error[i] > config.landmark_L1_outlier_thresh*2:
                    keep_x1[i] = 0
        else:
            keep_x1 += 1
        self.assign(out_data[1], req[1], mx.nd.array(keep_x1))
        self.assign(out_data[11], req[11], mx.nd.array(keep_x1))
		
        # x2,y2
        self.assign(out_data[2], req[2], in_data[1])
        self.assign(out_data[12], req[12], in_data[6])
        keep_x2 = np.zeros(pred_x2.shape[0])
        num = keep_x2.shape[0]
        if self.landmark_ohem:
            keep_num = int(len(keep_x2) * self.landmark_ohem_ratio)
            L1_error_x = abs(pred_x2 - target_x2)
            L1_error_y = abs(pred_y2 - target_y2)
            L1_error = L1_error_x+L1_error_y
            keep = np.argsort(L1_error.flat)[::-1][:keep_num]
            keep_x2[keep] = 1
            for i in range(num):
                #print L1_error[i]
                if L1_error[i] < config.landmark_L1_thresh*2 or L1_error[i] > config.landmark_L1_outlier_thresh*2:
                    keep_x2[i] = 0
        else:
            keep_x2 += 1
        self.assign(out_data[3], req[3], mx.nd.array(keep_x2))
        self.assign(out_data[13], req[13], mx.nd.array(keep_x2))
		
        # x3,y3
        self.assign(out_data[4], req[4], in_data[2])
        self.assign(out_data[14], req[14], in_data[7])
        keep_x3 = np.zeros(pred_x3.shape[0])
        num = keep_x3.shape[0]
        if self.landmark_ohem:
            keep_num = int(len(keep_x3) * self.landmark_ohem_ratio)
            L1_error_x = abs(pred_x3 - target_x3)
            L1_error_y = abs(pred_y3 - target_y3)
            L1_error = L1_error_x+L1_error_y
            keep = np.argsort(L1_error.flat)[::-1][:keep_num]
            keep_x3[keep] = 1
            for i in range(num):
                #print L1_error[i]
                if L1_error[i] < config.landmark_L1_thresh*2 or L1_error[i] > config.landmark_L1_outlier_thresh*2:
                    keep_x3[i] = 0
        else:
            keep_x3 += 1
        self.assign(out_data[5], req[5], mx.nd.array(keep_x3))
        self.assign(out_data[15], req[15], mx.nd.array(keep_x3))
		
        # x4,y4
        self.assign(out_data[6], req[6], in_data[3])
        self.assign(out_data[16], req[16], in_data[8])
        keep_x4 = np.zeros(pred_x4.shape[0])
        num = keep_x4.shape[0]
        if self.landmark_ohem:
            keep_num = int(len(keep_x4) * self.landmark_ohem_ratio)
            L1_error_x = abs(pred_x4 - target_x4)
            L1_error_y = abs(pred_y4 - target_y4)
            L1_error = L1_error_x+L1_error_y
            keep = np.argsort(L1_error.flat)[::-1][:keep_num]
            keep_x4[keep] = 1
            for i in range(num):
                #print L1_error[i]
                if L1_error[i] < config.landmark_L1_thresh*2 or L1_error[i] > config.landmark_L1_outlier_thresh*2:
                    keep_x4[i] = 0
        else:
            keep_x4 += 1
        self.assign(out_data[7], req[7], mx.nd.array(keep_x4))
        self.assign(out_data[17], req[17], mx.nd.array(keep_x4))
		
        # x5,y5
        self.assign(out_data[8], req[8], in_data[4])
        self.assign(out_data[18], req[18], in_data[9])
        keep_x5 = np.zeros(pred_x5.shape[0])
        num = keep_x5.shape[0]
        if self.landmark_ohem:
            keep_num = int(len(keep_x5) * self.landmark_ohem_ratio)
            L1_error_x = abs(pred_x5 - target_x5)
            L1_error_y = abs(pred_y5 - target_y5)
            L1_error = L1_error_x+L1_error_y
            keep = np.argsort(L1_error.flat)[::-1][:keep_num]
            keep_x5[keep] = 1
            for i in range(num):
                #print L1_error[i]
                if L1_error[i] < config.landmark_L1_thresh*2 or L1_error[i] > config.landmark_L1_outlier_thresh*2:
                    keep_x5[i] = 0
        else:
            keep_x5 += 1
        self.assign(out_data[9], req[9], mx.nd.array(keep_x5))
        self.assign(out_data[19], req[19], mx.nd.array(keep_x5))
		
		
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(10):
            keep = out_data[i*2+1].asnumpy().reshape(-1, 1)
            grad = keep / len(np.where(keep == 1)[0])
            self.assign(in_grad[i], req[i], mx.nd.array(grad))


@mx.operator.register("negativemining_onlylandmark10")
class NegativeMiningProp_OnlyLandmark10(mx.operator.CustomOpProp):
    def __init__(self):
        super(NegativeMiningProp_OnlyLandmark10, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['pred_x1','pred_x2','pred_x3','pred_x4','pred_x5','pred_y1','pred_y2','pred_y3','pred_y4','pred_y5', 
               'target_x1','target_x2','target_x3','target_x4','target_x5','target_y1','target_y2','target_y3','target_y4','target_y5']

    def list_outputs(self):
        return ['out_x1', 'keep_x1','out_x2', 'keep_x2','out_x3', 'keep_x3','out_x4', 'keep_x4','out_x5', 'keep_x5',
                'out_y1', 'keep_y1','out_y2', 'keep_y2','out_y3', 'keep_y3','out_y4', 'keep_y4','out_y5', 'keep_y5']

    def infer_shape(self, in_shape):
        #print(in_shape)
        keep_shape = (in_shape[0][0], )
        return in_shape, [in_shape[0], keep_shape,in_shape[0], keep_shape,in_shape[0], keep_shape,in_shape[0], keep_shape,in_shape[0], keep_shape,
                          in_shape[0], keep_shape,in_shape[0], keep_shape,in_shape[0], keep_shape,in_shape[0], keep_shape,in_shape[0], keep_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return NegativeMiningOperator_OnlyLandmark10()
