# coding: utf-8
"""
模型转换 ckpt > pb
"""
import os

import tensorflow as tf

from local_utils import data_utils
from tools import pred

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
    g = tf.Graph()
    with g.as_default():
        # 获取字符库
        charset = data_utils.get_charset("../../charset.3770.txt")
        decodes, prob, inputdata, batch_size = pred.build_graph(g, charset, 1)

        saver = tf.train.Saver()
        session = tf.Session(graph=g)
        saver.restore(sess=session, save_path=ckptModPath)

        # 保存转换训练好的模型
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelDir)

        inputs = {
            "inputdata": tf.saved_model.utils.build_tensor_info(inputdata),
            "batch_size": tf.saved_model.utils.build_tensor_info(batch_size),
        }
        output = {
            "decodes": tf.saved_model.utils.build_tensor_info(decodes[0]),
            "prob": tf.saved_model.utils.build_tensor_info(prob),
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
