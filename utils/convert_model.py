# coding: utf-8
"""
模型转换 ckpt > pb
"""
import os

import tensorflow as tf

from config import config
from crnn_model import crnn_model
from local_utils import data_utils

tf.app.flags.DEFINE_boolean('debug', True, '')
tf.app.flags.DEFINE_string('ckpt_mod_path', "", '')
tf.app.flags.DEFINE_string('charset', "../../charset.3770.txt", '')
tf.app.flags.DEFINE_string('save_mod_dir', "./model/crnn", '')

FLAGS = tf.app.flags.FLAGS


def convert():
    # 保存转换好的模型目录
    saveModDir = FLAGS.save_mod_dir

    # 每次转换都生成一个版本目录
    for i in range(100000, 9999999):
        cur = os.path.join(saveModDir, str(i))
        if not tf.gfile.Exists(cur):
            saveModDir = cur
            break

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    ses_config = tf.ConfigProto(gpu_options=gpu_options)
    print("模型保存目录", saveModDir)
    # 原ckpt模型
    ckptModPath = FLAGS.ckpt_mod_path
    print("CKPT模型目录", ckptModPath)
    print("charset目录", FLAGS.charset)
    # 获取字符库
    charset = data_utils.get_charset(FLAGS.charset)

    # 定义张量
    input_image = tf.placeholder(tf.float32, shape=[None, 32, None, 3], name='input_image')
    sequence_size = tf.placeholder(tf.int32, shape=[None])

    # 创建模型
    network = crnn_model.ShadowNet(phase='Train',
                                   hidden_nums=config.HIDDEN_UNITS,  # 256
                                   layers_nums=config.HIDDEN_LAYERS,  # 2层
                                   num_classes=len(charset))
    with tf.variable_scope('shadow', reuse=False):
        net_out,_ = network.build(inputdata=input_image, sequence_len=sequence_size)
    # 创建校验用的decode和编辑距离
    decoded = network.validate(net_out, sequence_size)

    saver = tf.train.Saver()
    session = tf.Session()
    saver.restore   (sess=session, save_path=ckptModPath)

    # 保存转换训练好的模型
    builder = tf.saved_model.builder.SavedModelBuilder(saveModDir)
    inputs = {
        "input_data": tf.saved_model.utils.build_tensor_info(input_image),
        "input_batch_size": tf.saved_model.utils.build_tensor_info(sequence_size),
    }
    # B.直接输出一个整个的SparseTensor
    output = {
        "output": tf.saved_model.utils.build_tensor_info(decoded),
    }

    # A.之前红岩的做法，他遇到的问题是，说无法直接绑定一个sparse_tensor，我测试貌似没事，我猜可能是由于tfserving版本的问题把，
    # 他用的是1.9，我用的是1.14
    # indices = decoded.indices
    # values = decoded.values
    # shape = decoded.dense_shape
    # output = {
    #     "output_indices": tf.saved_model.utils.build_tensor_info(indices),
    #     "output_values": tf.saved_model.utils.build_tensor_info(values),
    #     "output_shape": tf.saved_model.utils.build_tensor_info(shape),
    # }

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=output,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        sess=session,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={ # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        }
    )
    builder.save()
    print("转换模型结束", saveModDir)


if __name__ == '__main__':
    convert()
