import unittest
import matplotlib.pyplot as plt
from local_utils import image_util
import cv2

class TestImageUtil(unittest.TestCase):

    def test_init(self):
        pass

    def test_resize_by_height_with_padding(self):
        image = cv2.imread("0.png")
        #如果小于则加padding
        target_image = image_util.resize_by_height_with_padding(image,32,256)
        self.assertEqual(target_image.shape,(32,256,3))
        cv2.imwrite("test/out/padding01.png",target_image)
        plt.imshow(target_image)
        plt.show()
        target_image = image_util.resize_by_height_with_padding(image,32,32)
        self.assertEqual(target_image.shape,(32,32,3))
        cv2.imwrite("test/out/padding02.png", target_image)
        plt.imshow(target_image)
        plt.show()
        #如果大于指定宽度则缩放
        target_image = image_util.resize_by_height_with_padding(image, 32, 128)
        self.assertEqual(target_image.shape, (32, 128, 3))
        cv2.imwrite("test/out/resize01.png", target_image)
        plt.imshow(target_image)
        plt.show()


if __name__ == '__main__':
    unittest.main()