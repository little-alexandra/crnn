import cv2,os,numpy as np,random
from skimage import exposure
from skimage import io
from skimage import util

def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def save_image(original_name,type,dst_image):
    prefix,subfix = os.path.splitext(original_name)
    f_name = os.path.join("../data/test_result",prefix+"_"+type+subfix)
    print("保存图像：",f_name)
    cv2.imwrite(f_name,dst_image)

def filer2d(image):
    k = random.choice([2,3])
    kernel = np.ones((k,k), np.float32) / 9
    return cv2.filter2D(image, -1, kernel)

def filer_avg(img):
    blur = cv2.blur(img, (3, 3))  # 模板大小为3*5, 模板的大小是可以设定的
    box = cv2.boxFilter(img, -1, (3, 3))
    return blur

def filter_gaussian(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)  # （5,5）表示的是卷积模板的大小，0表示的是沿x与y方向上的标准差
    return blur

def filter_median(img):
    blur = cv2.medianBlur(img, 3)  # 中值滤波函数
    return blur

def filter_bi(img):
    blur = cv2.bilateralFilter(img, 5, 40, 40)
    return blur

# 曝光处理：参考：https://blog.csdn.net/limiyudianzi/article/details/86980680

def hist(image):
    dst_image = exposure.equalize_hist(image)
    # print(dst_image)
    return np.array(dst_image*255,dtype= np.uint8)

def adapthist(image):
    dst_image = exposure.equalize_adapthist(image)
    return np.array(dst_image*255,dtype= np.uint8)

def gamma(image):
    return exposure.adjust_gamma(image, gamma=0.5, gain=1)


def sigmoid(image):
    return exposure.adjust_sigmoid(image)

# 噪音函数： https://blog.csdn.net/weixin_44457013/article/details/88544918
def noise_gaussian(image):
    noise_gs_img = util.random_noise(image,mode='gaussian')
    # 靠，util.random_noise()返回的是[0,1]之间，所以要乘以255
    return np.array(noise_gs_img*255, dtype = np.uint8)

def noise_salt(image):
    noise_salt_img = util.random_noise(image,mode='salt')
    return np.array(noise_salt_img * 255, dtype=np.uint8)

def noise_pepper(image):
    temp = util.random_noise(image,mode='pepper')
    return np.array(temp*255, dtype = np.uint8)

def noise_sp(image):
    temp = util.random_noise(image,mode='s&p')
    return np.array(temp * 255, dtype=np.uint8)

def noise_speckle(image):
    temp = util.random_noise(image,mode='speckle')
    return np.array(temp * 255, dtype=np.uint8)

def adaptive_threshold(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯滤波
    # 自适应阈值化处理
    # cv2.ADAPTIVE_THRESH_MEAN_C：计算邻域均值作为阈值
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    return thresh
    #cv2.imshow("Mean Thresh", thresh)
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C：计算邻域加权平均作为阈值
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)


#dimming
def darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy

def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy

NOISE_MIN_NUM=10
NOISE_MAX_NUM=20

def noise(img):
    for i in range(20): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = np.random.randint(255)
    return img

# 腐蚀和膨胀：https://blog.csdn.net/hjxu2016/article/details/77837765

def erode(img):
    kernel_size = random.choice([(1,2),(2,1),(2,2)])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernel_size)
    img = cv2.erode(img,kernel)
    return img

def dilate(img):
    kernel_size = random.choice([(1, 2), (2, 1), (2, 2)])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernel_size)
    img = cv2.dilate(img,kernel)
    return img

enhance_method = [
    {"name": "2D", 'fun': filer2d},
    {"name": "均值", 'fun': filer_avg},
    {"name": "高斯", 'fun': filter_gaussian},
    {"name": "中值", 'fun': filter_median},
    {"name": "双边", 'fun': filter_bi},
    {"name": "直方图", 'fun': hist},
    {"name": "自适应直方图", 'fun': adapthist},
    {"name": "gamma", 'fun': gamma},
    {"name": "sigmod", 'fun': sigmoid},
    {"name": "锐化", 'fun': sharpen},
    {"name": "高斯噪音", 'fun': noise_gaussian},
    {"name": "盐噪音", 'fun': noise_salt},
    {"name": "胡椒噪音", 'fun': noise_pepper},
    {"name": "SP噪音", 'fun': noise_sp},
    {"name": "speckle噪音", 'fun': noise_speckle},
    {"name": "腐蚀", 'fun': erode},
    # {"name": "膨胀", 'fun': dilate},
    {"name": "噪音", 'fun': noise},
    {"name": "变暗", 'fun': darker},
    # {"name": "变亮", 'fun': brighter} #没啥用，反倒丢了纸质的特征

]


# 测试用
enhance_method = [
    {"name": "膨胀", 'fun': dilate}
]


def enhance(img):
    method = random.choice(enhance_method)
    dst_image = method['fun'](img)
    print("图像增强：", method['name'])
    return dst_image


def enhance_all_with_save(img,f_name):
    for method in enhance_method:
        print("图像增强：",method['name'])
        dst_image = method['fun'](img)
        save_image(f_name, method['name'], dst_image)

def enhance_all(img):
    dst_images = []
    for method in enhance_method:
        print("图像增强：",method['name'])
        dst_images.append( method['fun'](img))
    return dst_images

def do_folder(folder):
    for f in os.listdir(folder):
        if not (os.path.exists("../data/test_result")):
            os.makedirs("../data/test_result")
        f_name = os.path.join(folder,f)
        img = cv2.imread(f_name)
        save_image(f,"原图",img)
        enhance_all(img,f)


def do_file(img_full_path):
    if not (os.path.exists("../data/test_result")): os.makedirs("../data/test_result")

    img = cv2.imread(img_full_path)

    _, f_name = os.path.split(img_full_path)
    save_image(f_name,"原图",img)
    enhance_all_with_save(img,f_name)


if __name__ == "__main__":
    # 处理一个文件
    # do_file("data/lena.png")

    # 处理一个目录
    #do_folder("data/bill")

    # do_folder("data/test")

    do_file("../data/test1.png")
    do_file("../data/test2.png")
    do_file("../data/test3.png")
    do_file("../data/test4.png")
    do_file("../data/test5.png")