#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Script
"""
import os
import tensorflow as tf
import time
import datetime
from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from config import config
from utils import tensor_util
from utils import image_util
from utils import text_util
from tools.early_stop import EarlyStop
from utils.data_factory import DataFactory

tf.app.flags.DEFINE_string( 'name', 'CRNN', 'no use ,just a flag for shell batch')
tf.app.flags.DEFINE_boolean('debug', False, 'debug mode')
tf.app.flags.DEFINE_string( 'train_dir','data/train','')
tf.app.flags.DEFINE_integer('train_batch',64,'')
tf.app.flags.DEFINE_integer('train_steps',1000000,'')
tf.app.flags.DEFINE_string( 'label_file','train.txt','')
tf.app.flags.DEFINE_string( 'charset','','')
tf.app.flags.DEFINE_string( 'tboard_dir', 'tboard', '')
tf.app.flags.DEFINE_string( 'weights_path', None, '')
tf.app.flags.DEFINE_integer('validate_steps', 10, '')
tf.app.flags.DEFINE_string( 'validate_file','data/test.txt','')
tf.app.flags.DEFINE_integer('validate_batch',8,'')
tf.app.flags.DEFINE_integer('num_threads', 4, '')
tf.app.flags.DEFINE_float(  'learning_rate',0.001,'')
tf.app.flags.DEFINE_integer('early_stop', 10, '')
FLAGS = tf.app.flags.FLAGS

logger = log_utils.init_logger()


def save_model(saver,sess,epoch):
    model_save_dir = 'model'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(model_save_dir, model_name)
    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    logger.info("训练: 保存了模型：%s", model_save_path)


def create_summary_writer(sess):
    # 按照日期，一天生成一个Summary/Tboard数据目录
    # Set tf summary
    if not os.path.exists(FLAGS.tboard_dir): os.makedirs(FLAGS.tboard_dir)
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = os.path.join(FLAGS.tboard_dir,today)
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary_writer.add_graph(sess.graph)
    return summary_writer


def train(weights_path=None):
    logger.info("开始训练")

    # 获取字符库
    characters = text_util.get_charset(FLAGS.charset)

    # 定义张量
    input_image = tf.placeholder(tf.float32, shape=[None, 32, None, 3], name='input_image')
    sparse_label = tf.sparse_placeholder(tf.int32)
    sequence_size = tf.placeholder(tf.int32, shape=[None])

    # 创建模型
    network = crnn_model.ShadowNet(phase='Train',
                                     hidden_nums=config.HIDDEN_UNITS, # 256
                                     layers_nums=config.HIDDEN_LAYERS,# 2层
                                     num_classes=len(characters) + 1)

    with tf.variable_scope('shadow', reuse=False):
        net_out = network.build(inputdata=input_image, sequence_len=sequence_size)


    # 创建优化器和损失函数的op
    cost, optimizer, global_step = network.loss(net_out, sparse_label, sequence_size)

    # 创建校验用的decode和编辑距离
    validate_decode, sequence_dist = network.validate(net_out, sparse_label, sequence_size)

    # 创建一个变量用于把计算的精确度加载到summary中
    sess = tf.Session()
    accuracy = tf.Variable(0, name='accuracy', trainable=False)
    tf.summary.scalar(name='accuracy', tensor=accuracy)
    train_summary_op    = tf.summary.merge_all(scope="train")
    validate_summary_op = tf.summary.merge_all(scope="validate")
    summary_writer = create_summary_writer(sess)
    saver = tf.train.Saver()
    logger.debug("创建session")

    from tools.early_stop import EarlyStop

    early_stop = EarlyStop(FLAGS.early_stop)

    with sess.as_default():

        sess.run(tf.local_variables_initializer())
        if weights_path is None:
            logger.info('从头开始训练，不加载旧模型')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('从文件{:s}恢复模型，继续训练'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        data_generator = DataFactory.get_batch(data_dir=FLAGS.train_dir,
                                               charsets=characters,
                                               data_type='train',
                                               batch_size=FLAGS.train_batch,
                                               num_workers=FLAGS.num_threads)
        for epoch in range(1, FLAGS.train_steps + 1):
            logger.info("训练: 第%d次，开始", epoch)

            data = next(data_generator)
            data_image = image_util.resize_batch_image(data[0],config.INPUT_SIZE)
            data_seq = [(img.shape[1] // config.WIDTH_REDUCE_TIMES) for img in data_image]
            data_label = tensor_util.to_sparse_tensor(data[1])

            # validate一下
            if epoch % FLAGS.validate_steps == 0:
                logger.info('此Epoch为检验(validate)')
                logger.debug("%r",validate_summary_op)
                seq_distance,preds,labels_sparse,v_summary = sess.run(
                    [sequence_dist, validate_decode, sparse_label, validate_summary_op],
                    feed_dict={ input_image:data_image,
                                sparse_label:tf.SparseTensorValue(data_label[0], data_label[1], data_label[2]),
                                sequence_size: data_seq })

                _accuracy = data_utils.caculate_accuracy(preds, labels_sparse,characters)
                tf.assign(accuracy, _accuracy) # 更新正确率变量
                logger.info('正确率计算完毕：%f', _accuracy)
                summary_writer.add_summary(summary=v_summary, global_step=epoch)
                if is_need_early_stop(early_stop,accuracy,saver,sess,epoch): break

            # 单纯训练
            else:
                _, ctc_lost, t_summary = sess.run([optimizer, cost, train_summary_op],
                    feed_dict={ input_image:data_image,
                                # input_label: data_label,
                                sparse_label:tf.SparseTensorValue(data_label[0], data_label[1], data_label[2]),
                                sequence_size: data_seq })
                summary_writer.add_summary(summary=t_summary, global_step=epoch)

            logger.info('训练: 第{:d}次，结束'.format(epoch))

    sess.close()



def is_need_early_stop(early_stop,f1_value,saver,sess,step):
    decision = early_stop.decide(f1_value)

    if decision == EarlyStop.ZERO: # 当前F1是0，啥也甭说了，继续训练
        return False

    if decision == EarlyStop.CONTINUE:
        logger.info("新F1值比最好的要小，继续训练...")
        return False

    if decision == EarlyStop.BEST:
        logger.info("新F1值[%f]大于过去最好的F1值，早停计数器重置，并保存模型", f1_value)
        save_model(saver, sess, step)
        return False

    if decision == EarlyStop.STOP:
        logger.warning("超过早停最大次数，也尝试了多次学习率Decay，无法在提高：第%d次，训练提前结束", step)
        return True

    logger.error("无法识别的EarlyStop结果：%r",decision)
    return True


if __name__ == '__main__':
    print("开始训练...")
    train()

