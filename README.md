## 6.25 piginzoo
重新启动造样本的工作，以及请同事们帮着打了一些标注文件

- 修改了data_generator，支持多进程同时生成，提高效率
- 增加了sample_enhance，用来基于标注文件来生成更多的样本
- 增加了image_enhance，实现了大于19种增强算法，替换了之前的3种
- 微调了image_enhance的各类参数，特别是kernel size
- 微调了ctpn中的文字生成，增加了特殊字符的增强，空格的增强


## 6.19 piginzoo
针对的问题：
-	小数点被识别成了逗号
-	空格被认成负号，原因是缺少左右空格的样本
-	英文效果不好
-	有些缺一角的识别不好
针对这些问题做了如下改进：
-	发现了一个bug，数字都是整数，所以导致没有小数点
-	对所有数字，会左右加空格
-	专门有一部分做字母生成，字母和数字混合，字母和数字和汉字混合在校验的时候，把不认识的字都替换成了■，并且把OolIZS=>0011zs
-	用哼仔提供的程序切除的500万张里面挑了2万张来做验证集

然后在服务器上做了学习

* 在6880的基础上做增强学习

专门调节生成概率，主要生成带空格数字、英文字母极其混合。50万张。来做增强训练（加载原来模型，继续训练）
在crnn目录，字符集用charset.6880.txt

* 训练3770的字符集

通用生成了100万张图片，字符集用3770的，在crnn目录，字符集用charset.3770.txt。


## 6.11 piginzoo
- 实现了一个早停，用的是编辑距离，因为是越来越小，所以用了一个负的编辑距离
- 实现了一个validate batch，1500张左右，用来计算编辑距离和正确率
- 正确率抛弃了繁琐的计算，规则简单粗暴，两个字符串一致才算正确
- 重构了validate，并且把训练和validate的scalar summary分开
- 重新生成了150万张图片用于训练

## 6.7 piginzoo
几个细节：
- 基于远航的版本改，不再用之前的方式，而是直白的方式加载数据
- 修改了训练的adam的leaning rate，从0.1变成了0.001，这个是adam默认的值
- 不需要考虑lable的padding了，直接转化成了SparseTensor了，所以不用padding了
- 因为不用padding为0，所以词表的第一个不用空出来了
- 不能用不定长的图像宽度，因为虽然dynamical-lstm支持，但是之前得过一个VGG，他要求是预定义统一宽度的
- 因此，也决定只能resize成统一，所以加了一个resize为中位数的，尽量减少形变
- 因为讨厌padding成0（黑色）或者255（白色），所以只选择resize
- 不过前面的CNN怎么也不能支持宽度吧，必须要resize把，突然意识到，所谓一个批次可以宽度不定，是指的RNN的sequence吧（突然很绝望）

我Fork了这个版本，主要目的是为加上注释，呵呵。

# CRNN_Tensorflow
Use tensorflow to implement a Deep Neural Network for scene text recognition mainly based on the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition".You can refer to their paper for details http://arxiv.org/abs/1507.05717. Thanks for the author [Baoguang Shi](https://github.com/bgshih).  
This model consists of a CNN stage, RNN stage and CTC loss for scene text recognition task.

## Installation

This software has mostly been tested on Ubuntu 16.04(x64) using python3.5 and tensorflow 1.3.0 with cuda-8.0, cudnn-6.0 and a GTX-1070 GPU. Other versions of tensorflow have not been tested but might work properly above version 1.0.

The following methods are provided to install dependencies:


## Docker

There are Dockerfiles inside the folder `docker`. Follow the instructions inside `docker/README.md` to build the images.

## Conda

You can create a conda environment with the required dependencies using: 

```
conda env create -f crnntf-env.yml
```

## Pip

Required packages may be installed with

```
pip3 install -r requirements.txt
```

## Test model
In this repo I uploaded a model trained on a subset of the [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). During data preparation process the dataset is converted into a tensorflow records which you can find in the data folder.
You can test the trained model on the converted dataset by

```
python tools/test_shadownet.py --dataset_dir data/ --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```
`Expected output is`  
![Test output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_output.png)
If you want to test a single image you can do it by
```
python tools/demo_shadownet.py --image_path data/test_images/test_01.jpg --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```
`Example image_01 is`  
![Example image1](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/text_example_image1.png)  
`Expected output_01 is`  
![Example image1 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image1_output.png)  
`Example image_02 is`  
![Example image2](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2.png)  
`Expected output_02 is`  
![Example image2 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2_output.png) 
`Example image_03 is`  
![Example image3](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese.png)  
`Expected output_03 is`  
![Example image3 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese_output.png)
`Example image_04 is`  
![Example image4](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/dmeo_chinese_2.png)  
`Expected output_04 is`  
![Example image4 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese_2_ouput.png)

## Train your own model
#### Data Preparation
Firstly you need to store all your image data in a root folder then you need to supply a txt file named sample.txt to specify the relative path to the image data dir and it's corresponding text label. For example

```
path/1/2/373_coley_14845.jpg coley
path/17/5/176_Nevadans_51437.jpg nevadans
```

Secondly you are supposed to convert your dataset into tensorflow records which can be done by
```
python tools/write_text_features --dataset_dir path/to/your/dataset --save_dir path/to/tfrecords_dir
```
All your training image will be scaled into (32, 100, 3) the dataset will be divided into train, test, validation set and you can change the parameter to control the ratio of them.

#### Train model
The whole training epoches are 40000 in my experiment. I trained the model with a batch size 32, initialized learning rate is 0.1 and decrease by multiply 0.1 every 10000 epochs. For more training parameters information you can check the global_configuration/config.py for details. To train your own model by

```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords
```
You can also continue the training process from the snapshot by
```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords --weights_path path/to/your/last/checkpoint
```
After several times of iteration you can check the log file in logs folder you are supposed to see the following contenent
![Training log](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_log.png)
The seq distance is computed by calculating the distance between two saparse tensor so the lower the accuracy value is the better the model performs.The train accuracy is computed by calculating the character-wise precision between the prediction and the ground truth so the higher the better the model performs.

During my experiment the `loss` drops as follows  
![Training loss](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_loss.png)
The `distance` between the ground truth and the prediction drops as follows  
![Sequence distance](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/seq_distance.png)

## Experiment
The accuracy during training process rises as follows  
![Training accuracy](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/training_accuracy.md)

## TODO
The model is trained on a subet of [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). So i will train a new model on the whold dataset to get a more robust model.The crnn model needs large of training data to get a rubust model.
