# coding: utf-8
"""
模型转换 ckpt > pb
"""
import os

import tensorflow as tf

from config import config
from crnn_model import crnn_model
from local_utils import data_utils

tf.app.flags.DEFINE_boolean('debug', False, '')
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')


def convert():
    # 保存转换好的模型目录
    savedModelDir = "./models/crnn"
    # 每次转换都生成一个版本目录
    for i in range(100000, 9999999):
        cur = os.path.join(savedModelDir, str(i))
        if not tf.gfile.Exists(cur):
            savedModelDir = cur
            break
    # 原ckpt模型
    ckptModPath = "../../../../models_ckpt/crnn_2019-06-27-05-06-48.ckpt-58000"
    ckptModPath = "../../model/crnn_2019-09-04-18-57-44.ckpt-4"
    # 获取字符库
    charset = data_utils.get_charset("../../charset.3770.txt")

    # 定义张量
    input_image = tf.placeholder(tf.float32, shape=[None, 32, None, 3], name='input_image')
    sequence_size = tf.placeholder(tf.int32, shape=[None])

    # 创建模型
    network = crnn_model.ShadowNet(phase='Train',
                                   hidden_nums=config.HIDDEN_UNITS,  # 256
                                   layers_nums=config.HIDDEN_LAYERS,  # 2层
                                   num_classes=len(charset) + 1)
    with tf.variable_scope('shadow', reuse=False):
        net_out = network.build(inputdata=input_image, sequence_len=sequence_size)
    # 创建校验用的decode和编辑距离
    validate_decode, shape, indices, values = network.validate(net_out, sequence_size)
    print(validate_decode)
    saver = tf.train.Saver()
    session = tf.Session()
    saver.restore(sess=session, save_path=ckptModPath)

    # 保存转换训练好的模型
    builder = tf.saved_model.builder.SavedModelBuilder(savedModelDir)
    inputs = {
        "input_data": tf.saved_model.utils.build_tensor_info(input_image),
        "input_batch_size": tf.saved_model.utils.build_tensor_info(sequence_size),
    }
    output = {
        "output_shape": tf.saved_model.utils.build_tensor_info(shape),
        "output_indices": tf.saved_model.utils.build_tensor_info(indices),
        "output_values": tf.saved_model.utils.build_tensor_info(values),
    }
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=output,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    builder.add_meta_graph_and_variables(
        session,
        [tf.saved_model.tag_constants.SERVING],
        {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        }
    )
    builder.save()


if __name__ == '__main__':
    convert()
