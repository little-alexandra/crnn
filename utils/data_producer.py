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

                image = image_util.read_image_file(image_file_name)
                label = data_utils.convert_label_to_id(_label, charsets)

                # 除了个bug，image加载失败了，所以为了防止这点，忽略空
                if label is None:
                    logger.error("解析标签字符串失败，忽略此样本：%s",_label)
                    continue
                if image is None:
                    logger.error("解析样本图片失败，忽略此样本：%s", image_file_name)
                    continue

                yield image, label



