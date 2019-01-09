train mtcnn: a modified version by Zuo Qing from https://github.com/Seanlinx/mtcnn

训练环境windows 7/10, 其他环境未测试

**六种Pnet20（原版Pnet等价计算量为43.7M）**

| 模型名称 | 输入尺寸 | cell_size | stride | 计算量（不计bbox）|
| -------- | ------   | --------- | -------| ------------      |
| Pnet20_v1| 320x240  | 20        |  4     | 11.6 M            |
| Pnet20_v2| 320x240  | 20        |  4     | 18.4 M            |
| Pnet20_v3| 320x240  | 20        |  4     | 28.1 M            |
| Pnet20_v4| 320x240  | 20        |  4     | 46.9 M            |
| Pnet20_v5| 320x240  | 20        |  4     | 74.1 M            |
| Pnet20_v6| 320x240  | 20        |  4     | 91.1 M            |

# 基本说明

**(1)请使用[ZQCNN_MTCNN](https://github.com/zuoqing1988/ZQCNN)来进行forward**

**(2)Pnet改为Pnet20需要在你的MTCNN中更改cell_size=20, stride=4**

	1920*1080图像找20脸，第一层Pnet20输入尺寸1920x1080，计算量347.9M，原版Pnet输入1152x648，计算量1278.0M

**(3)Rnet保持size=24不变，网络结构变为dw+sep，计算量约为原版1/3**

**(4)Onet带landmark我没有训练成功过**

**(5)Lnet是专门训练landmark的**

# 训练建议

**(1)下载[WIDER_train](https://pan.baidu.com/s/1PSR11Xs8lWmtVazCGoYR7Q)解压到data文件夹**

	解压之后目录为data/WIDER_train/images

**(2)双击gen_anno_and_train_list.bat**

	生成prepare_data/wider_annotations/anno.txt和data/mtcnn/imglists/train.txt

## 训练Pnet20 

**(3)双击P20_gen_data.bat**

	生成训练Pnet20所需样本
	
**(4)双击P20_gen_imglist.bat**

	生成训练Pnet20的list文件

**(5)双击P20_train.bat**

	训练Pnet20
	
**(6)双击P20_gen_hard_example.bat**

	利用训练得到的Pnet20模型，生成用于进一步训练Pnet20的hard样本，请用文本方式打开，酌情填写参数
		
**(7)双击P20_gen_imglist_with_hard.bat**

	生成用于进一步训练Pnet20的list文件
	
**(8)双击P20_train_with_hard.bat**
	
	进一步训练Pnet20
	
## 训练Rnet

**(9)双击R_gen_data.bat**

	生成训练Rnet所需样本
	
**(10)双击R_gen_hard_example.bat**
	
	利用训练得到的Pnet20模型，生成用于训练Rnet的hard样本，请用文本方式打开，酌情填写参数
	
**(11)双击R_gen_imglist_with_hard.bat**

	生成用于训练Rnet的list文件
	
**(12)双击R_train_with_hard.bat**

	训练Rnet
	
## 训练Onet

**(13)双击O_gen_data.bat**

	生成训练Onet所需样本
	
**(14)双击O_gen_hard_example.bat**
	
	利用训练得到的Pnet20、Rnet模型，生成用于训练Onet的hard样本，请用文本方式打开，酌情填写参数
	
**(15)双击O_gen_imglist_with_hard.bat**

	生成用于训练Onet的list文件
	
## 不带landmark
**(16)双击O_train_with_hard.bat**

	训练Onet
	
## 带landmark

下载[img_cut_celeba](https://pan.baidu.com/s/1XeGsYT_6VCP3n177oa3KGw)，解压到data/img_cut_celeba

图片位置在data/img_cut_celeba/xx.jpg

**(17)双击O_gen_landmark.bat**

	生成训练Onet所需landmark样本

**(18)双击O_gen_imglist_with_hard_landmark.bat**

	生成用于训练Onet的list文件

## 单独训练landmark

**(19)双击L_train.bat**

	训练Lnet
	
# 省硬盘的方式训练landmark

选择以下三个数据集之一：(A)是原始celeba数据，(B)(C)是我加工过的，加载速度B>C>A，（**我推荐用C，理论上用C应该和用A训练出来的结果一样**）

(A)[img_celeba](https://pan.baidu.com/s/1f6lYVNVYR7h28Vh-1nIPnQ)，解压到data/img_celeba

图片位置在data/img_celeba/xx.jpg

以文本方式编辑 L_train_onlylandmark.bat, 设置参数--image_set img_celeba_all

修改config.py中config.landmark_img_set='img_celeba'

双击 L_train_onlylandmark.bat 运行

(B)[img_align_celeba](https://pan.baidu.com/s/1rUBW8NasLZGtfQ33uA6Kdg)，解压到data/img_align_celeba

图片位置在data/img_align_celeba/xx.jpg

以文本方式编辑 L_train_onlylandmark.bat, 设置参数--image_set img_align_celeba_good

修改config.py中config.landmark_img_set='img_align_celeba'

双击 L_train_onlylandmark.bat 运行

(C)[img_cut_celeba](https://pan.baidu.com/s/1XeGsYT_6VCP3n177oa3KGw)，解压到data/img_cut_celeba

图片位置在data/img_cut_celeba/xx.jpg

以文本方式编辑 L_train_onlylandmark.bat, 设置参数--image_set img_cut_celeba_all

修改config.py中config.landmark_img_set='img_cut_celeba'

双击 L_train_onlylandmark.bat 运行

**备注：调整minibatch_onlylandmark.py里的参数得到的landmark精度不一样**

# 训练106点landmark

**暂不提供关键脚本minibatch_onlylandmark106.py**

下载[Training_data106](https://pan.baidu.com/s/1SCdyksAWRSvhWCOJ4Syk1A)解压到data/Training_data106

解压后目录结构应为

	data/Training_data106/AFW
	data/Training_data106/HELEN
	data/Training_data106/IBUG
	data/Training_data106/LFPW
	
将 data/Training_data106/landmark106.txt拷贝到data/mtcnn/imglists/landmark106.txt， 在config.py设置

	config.landmark_img_set = 'Training_data106'
	
双击L_train_onlylandmark106.bat开始训练

