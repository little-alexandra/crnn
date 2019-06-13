import cv2,numpy as np
image1 = cv2.imread("0.png")
image2 = cv2.imread("1.png")
images = [image1,image2]
image_list = np.array(images)
print(image_list.shape)
