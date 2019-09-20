from crnn.tools import pred
import cv2,os
from crnn.config import config
from crnn.local_utils import log_utils, data_utils
import tensorflow as tf
import random

logger = log_utils.init_logger()
FLAGS = tf.app.flags.FLAGS




def validate():

    charset  = data_utils.get_charset(FLAGS.charset)
    image_names, labels = data_utils.read_labeled_image_list(FLAGS.label_file,charset)
    labels = data_utils.process_unknown_charactors_all(labels,charset,replace_char='■')

    # 遍历图片目
    logger.debug("加载目录[%s]下的所有图片[%d]张", FLAGS.image_dir,len(image_names))
    image_names_labels = list(zip(image_names,labels))
    if len(image_names)>FLAGS.validate_num:
        image_names_labels = random.sample(image_names_labels,FLAGS.validate_num)
    logger.debug("筛选出%d张用于验证",len(image_names_labels))

    # 加载图片
    image_list = []
    labels = []
    for image_name,label in image_names_labels:
        _,subfix = os.path.splitext(image_name)
        if subfix.lower() not in ['.jpg','.png','.jpeg','.gif','.bmp']: continue
        image = cv2.imread(image_name)

        if not os.path.exists(image_name):
            logger.error("需要验证的图片[%s]不存在",image_name)
            continue

        logger.debug("加载图片:%s", image_name)
        image_list.append(image)
        labels.append(label)


    sess = pred.initialize()

    pred_result,_ = pred.pred(image_list,
                              config.cfg.TRAIN.VAL_BATCH_SIZE,
                              sess)

    return pred_result,labels


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('crnn_model_dir', "model",'')
    tf.app.flags.DEFINE_string('crnn_model_file', None,'')
    tf.app.flags.DEFINE_boolean('debug', True,'')
    tf.app.flags.DEFINE_string('charset','charset6k.txt', '')
    tf.app.flags.DEFINE_string('image_dir', 'data/test','')
    tf.app.flags.DEFINE_string('label_file','data/test.txt', '')
    tf.app.flags.DEFINE_integer('validate_num', 1000,'')

    pred_result, labels = validate()
    p = data_utils.caculate_accuracy(pred_result, labels)
    logger.info("验证了%d条数据，正确率为：%f",len(labels),p)