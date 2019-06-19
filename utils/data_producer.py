import logging
import os
import numpy as np
from local_utils import image_util
from local_utils import data_utils

logger = logging.getLogger("data producer")


class DataProducer:

    @staticmethod
    def work(label_file_name, charsets,unknow_char=None):
        image_file_names, labels = data_utils.read_labeled_image_list(label_file_name,charsets,unknow_char)
        image_labels = list(zip(image_file_names, labels))
        while True:
            np.random.shuffle(image_labels)
            for image_file_name,_label in image_labels:

                if not os.path.exists(image_file_name):
                    logger.warning("标签文件[%s]不存在啊",image_file_name)
                    continue

                processed_label = data_utils.process_unknown_charactors(_label, charsets, unknow_char)
                if processed_label is None or len(processed_label)==0:
                    logger.error("解析标签字符串失败，忽略此样本：[%s]",_label)
                    continue

                label_index = data_utils.convert_label_to_id(processed_label, charsets)
                if label_index is None: continue

                image = image_util.read_image_file(image_file_name)
                if image is None:
                    logger.error("解析样本图片失败，忽略此样本：[%s]", image_file_name)
                    continue

                yield image, label_index

            logger.info("遍历完所有的样本，继续下一个epochs遍历")

