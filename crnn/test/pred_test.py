import unittest

from local_utils import image_util
import cv2
from tools import pred
class PredUtil(unittest.TestCase):

    def test_init(self):
        pass

    def test_get_latest_model(self):
        model_name = pred.get_latest_model("model")
        self.assertIsNotNone(model_name)
        print("最新的Model为：[%s]" % model_name)

if __name__ == '__main__':
    unittest.main()