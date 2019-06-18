import logging
import os
import numpy as np
from local_utils import image_util
from local_utils import data_utils

logger = logging.getLogger("data producer")


class DataProducer:

    @staticmethod
    def work(label_file_name, charsets,unknow_char=None):
        image_file_names, labels = data_utils.read_labeled_image_list(label_file_name,charsets)

        image_labels = list(zip(image_file_names, labels))
        while True:
            np.random.shuffle(image_labels)
            for image_file_name,label in image_labels:

                if not os.path.exists(image_file_name):
                    logger.warning("标签文件[%s]不存在啊",image_file_name)
                    continue

                image = image_util.read_image_file(image_file_name)
                label = data_utils.convert_label_to_id(label, charsets)

                if label is None:continue

                yield image, label



