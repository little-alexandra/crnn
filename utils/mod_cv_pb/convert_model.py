# coding: utf-8
"""
模型转换 ckpt > pb
"""
import os

import tensorflow as tf

from config import config
from crnn_model import crnn_model  # 自己的模型网络
from local_utils import data_utils

tf.app.flags.DEFINE_boolean('debug', False, '')

def convert():
    __rootPath = "../../"
    # 保存转换好的模型目录
    savedModelDir = __rootPath + "../../ai/models/crnn"
    # 每次转换都生成一个版本目录
    for i in range(100000, 9999999):
        cur = os.path.join(savedModelDir, str(i))
        if not tf.gfile.Exists(cur):
            savedModelDir = cur
            break

    # 原ckpt模型
    ckptModPath = __rootPath + "../../ai/models_ckpt/crnn_2019-06-27-05-06-48.ckpt-58000"
    # 获取字符库
    charset = data_utils.get_charset(__rootPath + "charset.3770.txt")
    # 输入张量
    input_image = tf.placeholder(tf.float32, shape=[None, 32, None, 3], name='input_image')
    sequence_size = tf.placeholder(tf.int32, shape=[None])
    # 创建模型
    network = crnn_model.ShadowNet(phase='Train',
                                   hidden_nums=config.HIDDEN_UNITS,  # 256
                                   layers_nums=config.HIDDEN_LAYERS,  # 2层
                                   num_classes=len(charset) + 1)

    with tf.variable_scope('shadow', reuse=False):
        net_out = network.build(inputdata=input_image, sequence_len=sequence_size)
    print(net_out)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=ckptModPath)

    # 保存转换训练好的模型
    builder = tf.saved_model.builder.SavedModelBuilder(savedModelDir)
    inputs = {
        # ph_input_image 就是模型里面定义的输入placeholder
        "input_image": tf.saved_model.utils.build_tensor_info(input_image),
        "sequence_size": tf.saved_model.utils.build_tensor_info(sequence_size)
    }
    # model > classes 是模型的输出， 预测的时候就是这个输出
    output = {
        "output": tf.saved_model.utils.build_tensor_info(net_out)
    }
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=output,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    builder.add_meta_graph_and_variables(
        session,
        [tf.saved_model.tag_constants.SERVING],
        {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
    )
    builder.save()


if __name__ == '__main__':
    convert()
