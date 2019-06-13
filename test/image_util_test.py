import unittest

from local_utils import image_util
import cv2

class TestImageUtil(unittest.TestCase):

    def test_init(self):
        pass

    def test_resize_by_height_with_padding(self):
        image = cv2.imread("test/0.png")

        target_image = image_util.resize_by_height_with_padding(image,32,512)
        self.assertEqual(target_image.shape,(32,512,3))
        cv2.imwrite("test/out/padding01.png",target_image)

        target_image = image_util.resize_by_height_with_padding(image,32,32)
        self.assertEqual(target_image.shape,(32,32,3))
        cv2.imwrite("test/out/padding02.png", target_image)


if __name__ == '__main__':
    unittest.main()