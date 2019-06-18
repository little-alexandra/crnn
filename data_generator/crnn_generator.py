import sys,os,random
from PIL import Image
sys.path.append("../ctpn")
from data_generator import generator
from local_utils import data_utils
import numpy as np

POSSIBILITY_CUT_EDGE = 0.1 # 10%的概率，会缺少个边
MAX_CUT_EDGE = 4           # 最大的边的缺的像素


DEBUG = True

'''
    用来生成crnn的样本，之前写的generator.py已经改名=>crnn_generator.py了，
    原因是ctpn做了类似的事了，就不生产duplicated code了，
    但是由于涉及到另外一个项目，所以在import的时候，可以看到"sys.path.append("../ctpn")"这样诡异代码，望理解，
    言外之意，就是ctpn项目得在上级目录，并且，叫做ctpn的文件夹名字
'''

# 生成一张图片
def create_backgroud_image(bground, width, height):
    if DEBUG: print("width, height: %d,%d" % (width, height))
    # 在大图上随机产生左上角
    x = random.randint(0,bground.size[0]-width)
    y = random.randint(0,bground.size[1]-height)
    bground = bground.crop((x, y, x+width, y+height))
    return bground

# 给定一个四边形，得到他的包裹的矩形的宽和高，用来做他的背景图片
def rectangle_w_h(points):
    # if DEBUG: print("points:%r",points)
    x_min,y_min = np.min(points,axis=0)
    x_max,y_max = np.max(points,axis=0)
    # if DEBUG: print("x_min:%f,x_max:%f,y_min:%f,y_max:%f" %(x_min,x_max,y_min,y_max))
    return int(x_max - x_min), int(y_max - y_min)

def main(save_path, num, label_file,charset,all_bg_images):

    words_image, _, _, random_word, points = generator.create_one_sentence_image(charset)
    # points,返回的是仿射的4个点，不是矩形，是旋转了的

    # 弄一个矩形框包裹他
    width , height = rectangle_w_h(points)

    # 生成一张背景图片，剪裁好
    background_image = create_backgroud_image(
                        random.choice(all_bg_images),
                        width,
                        height)

    offset = 0
    if np.random.choice([True, False], p=[POSSIBILITY_CUT_EDGE, 1 - POSSIBILITY_CUT_EDGE]):
        offset = random.randint(-MAX_CUT_EDGE,MAX_CUT_EDGE)

    background_image.paste(words_image, (0,offset), words_image)

    # 保存文本信息和对应图片名称
    image_file_name = str(num) + '.png'
    save_path = os.path.join(save_path, image_file_name)
    if DEBUG: print("文件名：%s" % save_path)

    label_file.write(save_path + " " + random_word + '\n')

    background_image.save(save_path)


# 注意：需要在根目录下运行，存到 /data/train目录下
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    parser.add_argument("--dir")
    parser.add_argument("--num")
    parser.add_argument("--charset")#"charset.6883.txt"

    args = parser.parse_args()

    DATA_DIR = args.dir
    TYPE= args.type
    charset_file_name= args.charset

    if not os.path.exists(args.charset):
        print("字符集文件[%s]不存在" % args.charset)
        exit(-1)
    if not os.path.exists(DATA_DIR):os.makedirs(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR,TYPE)): os.makedirs(os.path.join(DATA_DIR,TYPE))


    # 同时生成label，记录下你生成标签文件名
    label_file_name = os.path.join(DATA_DIR,TYPE+".txt")
    label_file = open(label_file_name, 'w', encoding='utf-8')
    total = int(args.num)

    # 加载字符集
    charset = data_utils.get_charset(charset_file_name)

    # 预先加载所有的纸张背景
    all_bg_images = generator.load_all_backgroud_images(os.path.join('../ctpn/data_generator/background/'))

    # 生成图片数据
    label_dir = os.path.join(DATA_DIR,TYPE)
    for num in range(0,total):
        main(label_dir, num, label_file,charset,all_bg_images)
        if DEBUG: print("--------------------")
        if num % 1000 == 0:
            print('生成了样本[%d/%d]'%(num,total))
