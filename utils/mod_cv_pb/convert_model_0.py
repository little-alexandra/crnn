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
    ckptModPath = "../../model/crnn_2019-09-04-16-07-59.ckpt-4"
    # 获取字符库
    charset = data_utils.get_charset("../../charset.3770.txt")

    # # 定义张量
    # input_image = tf.placeholder(tf.float32, shape=[None, 32, None, 3], name='input_image')
    # sparse_label = tf.sparse_placeholder(tf.int32)
    # sequence_size = tf.placeholder(tf.int32, shape=[None])
    #
    # # 创建模型
    # network = crnn_model.ShadowNet(phase='Train',
    #                                hidden_nums=config.HIDDEN_UNITS,  # 256
    #                                layers_nums=config.HIDDEN_LAYERS,  # 2层
    #                                num_classes=len(charset) + 1)
    #
    # with tf.variable_scope('shadow', reuse=False):
    #     net_out = network.build(inputdata=input_image, sequence_len=sequence_size)
    #
    # # 创建优化器和损失函数的op
    # cost, _ = network.loss(net_out, sparse_label, sequence_size)

    g = tf.Graph()
    with g.as_default():
        decodes, prob, inputdata, batch_size = pred.build_graph(g, charset, 1)
        saver = tf.train.Saver()
        session = tf.Session(graph=g)
        saver.restore(sess=session, save_path=ckptModPath)
        # t0=session.graph.get_tensor_by_name("CTCBeamSearchDecoder:0")
        # print(t0)
        # t0=session.graph.get_tensor_by_name("CTCBeamSearchDecoder:1")
        # print(t0)
        # t0=session.graph.get_tensor_by_name("CTCBeamSearchDecoder:2")
        # print(t0)

        op0 = session.graph.get_operation_by_name("CTCBeamSearchDecoder")
        print(op0)
        session.graph.get_tensor_by_name()
        # print(session.graph.get_operations())
        # for op in session.graph.get_operations():
        #     print(op)
        # for node in session.graph_def.node:
        #     print("--------  ",node)
        # 保存转换训练好的模型
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelDir)

        inputs = {
            "inputdata": tf.saved_model.utils.build_tensor_info(inputdata),
            "batch_size": tf.saved_model.utils.build_tensor_info(batch_size),
        }

        # indices = decodes.indices
        # values = decodes.values
        # dense_shape = decodes.dense_shape
        # indices_tensor_proto = tf.saved_model.utils.build_tensor_info(indices)
        # values_tensor_proto = tf.saved_model.utils.build_tensor_info(values)
        # dense_shape_tensor_proto = tf.saved_model.utils.build_tensor_info(dense_shape)
        # print(decodes.indices.name)
        # print(decodes.values.name)
        # print(decodes.dense_shape.name)
        # print(indices_tensor_proto)
        output = {
            "decodes": tf.saved_model.utils.build_tensor_info(op0[0]),
            "prob": tf.saved_model.utils.build_tensor_info(prob)
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
