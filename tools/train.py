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
tf.app.flags.DEFINE_string( 'data_dir','data/train','')
tf.app.flags.DEFINE_integer('train_batch',64,'')
tf.app.flags.DEFINE_integer('train_steps',1000000,'')
tf.app.flags.DEFINE_integer('train_num_threads', 4, '')
tf.app.flags.DEFINE_string( 'label_file','train.txt','')
tf.app.flags.DEFINE_string( 'charset','','')
tf.app.flags.DEFINE_string( 'tboard_dir', 'tboard', '')
tf.app.flags.DEFINE_string( 'weights_path', None, '')
tf.app.flags.DEFINE_string( 'validate_file','data/test.txt','')
tf.app.flags.DEFINE_integer('validate_batch',8,'')
tf.app.flags.DEFINE_integer('validate_steps', 10, '')
tf.app.flags.DEFINE_integer('validate_num', 1000, '')
tf.app.flags.DEFINE_integer('validate_num_threads', 1, '')
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
    cost, optimizer = network.loss(net_out, sparse_label, sequence_size)

    # 创建校验用的decode和编辑距离
    validate_decode = network.validate(net_out, sequence_size)

    # 创建一个变量用于把计算的精确度加载到summary中
    accuracy = tf.Variable(0, name='accuracy', dtype=tf.float32,trainable=False)
    edit_distance = tf.Variable(0, name='edit_distance', dtype=tf.float32, trainable=False)
    tf.summary.scalar(name='edit_distance', tensor=edit_distance)  # 这个只是看错的有多离谱，并没有当做损失函数，CTC loss才是核心
    tf.summary.scalar(name='accuracy', tensor=accuracy)
    summary_op    = tf.summary.merge_all()

    sess = tf.Session()
    summary_writer = create_summary_writer(sess)

    saver = tf.train.Saver()
    logger.debug("创建session")

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

        train_data_generator = DataFactory.get_batch(data_dir=FLAGS.data_dir,
                                               charsets=characters,
                                               data_type='train',
                                               batch_size=FLAGS.train_batch,
                                               num_workers=FLAGS.train_num_threads)

        validate_data_generator = DataFactory.get_batch(data_dir=FLAGS.data_dir,
                                               charsets=characters,
                                               data_type='validate',
                                               batch_size=FLAGS.validate_batch,
                                               num_workers=FLAGS.validate_num_threads)


        for epoch in range(1, FLAGS.train_steps + 1):
            logger.info("训练: 第%d次，开始", epoch)

            input_image_list,input_labels = next(train_data_generator)
            data_images = image_util.resize_batch_image(input_image_list,config.INPUT_SIZE)
            data_seq = [(img.shape[1] // config.WIDTH_REDUCE_TIMES) for img in data_images]
            data_labels_indices, data_labels_values, data_labels_shape = \
                tensor_util.to_sparse_tensor(input_labels)

            # validate一下
            if epoch % FLAGS.validate_steps == 0:
                _edit_distance = validate(accuracy,
                                         characters,
                                         edit_distance,
                                         input_image,
                                         sequence_size,
                                         sess,
                                         validate_data_generator,
                                         validate_decode)
                if is_need_early_stop(early_stop,_edit_distance,saver,sess,epoch): break

            _, ctc_lost, summary = sess.run([optimizer, cost, summary_op],
                feed_dict={ input_image:data_images,
                            sparse_label:tf.SparseTensorValue(data_labels_indices, data_labels_values, data_labels_shape),
                            sequence_size: data_seq })

            summary_writer.add_summary(summary=summary, global_step=epoch)

            logger.info('训练: 第{:d}次，结束'.format(epoch))

    sess.close()


def validate(accuracy, characters, edit_distance, input_image, sequence_size, sess, validate_data_generator, validate_decode):
    logger.info('Epoch为检验(validate)，开始，校验%d个样本',FLAGS.validate_num * FLAGS.validate_batch)
    labels = []
    preds = []
    start = time.time()
    for val_step in range(0, FLAGS.validate_num):
        input_image_list, input_labels = next(validate_data_generator)
        data_images = image_util.resize_batch_image(input_image_list, config.INPUT_SIZE)
        data_seq = [(img.shape[1] // config.WIDTH_REDUCE_TIMES) for img in data_images]
        preds_sparse = sess.run(validate_decode, feed_dict={input_image: data_images, sequence_size: data_seq})
        logger.debug("Validate Inference完毕，识别了%d张图片",len(data_images))
        _preds = data_utils.sparse_tensor_to_str(preds_sparse[0], characters)
        preds += _preds
        labels += data_utils.id2str(input_labels, characters)

    _accuracy = data_utils.caculate_accuracy(preds, labels)
    _edit_distance = data_utils.caculate_edit_distance(preds, labels)
    sess.run([tf.assign(accuracy, _accuracy), tf.assign(edit_distance, _edit_distance)])
    logger.info("Validate %d张样本和%d预测计算结果：正确率 %f,编辑距离 %f", _accuracy, _edit_distance)
    logger.info('Epoch检验(validate)结束，耗时：%d 秒', time.time() - start)
    return _edit_distance


def is_need_early_stop(early_stop,value,saver,sess,step):
    decision = early_stop.decide(value)

    if decision == EarlyStop.ZERO: # 当前Value是0，啥也甭说了，继续训练
        return False

    if decision == EarlyStop.CONTINUE:
        logger.info("新Value值比最好的要小，继续训练...")
        return False

    if decision == EarlyStop.BEST:
        logger.info("新Value值[%f]大于过去最好的Value值，早停计数器重置，并保存模型", value)
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

