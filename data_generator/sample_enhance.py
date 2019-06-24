# 对已有的数据做增强

import data_generator.image_enhance as enhancer
import os,cv2
# 遍历label文件，对每一个文件，挨个施加增强算法，产生对应的文件，并且生成新的label

def enhance(image,dir,image_name,label_file,label):
    name,ext = os.path.splitext(image_name)
    enhanced_images = enhancer.enhance_all(image)

    for i,enhanced_image in enumerate(enhanced_images):
        img_full_path = os.path.join(dir,name+"-"+str(i+1)+ext)
        print(img_full_path)
        cv2.imwrite(img_full_path,enhanced_image)
        label_file.write(img_full_path+" "+label+"\n")

    # 原图也保留一个
    img_full_path = os.path.join(dir, image_name)
    cv2.imwrite(img_full_path, image)
    label_file.write(img_full_path + " " + label + "\n")


def process(original_file_name,new_label_name,enhance_dir):
    new_label_file = open(new_label_name, 'w', encoding='utf-8')

    original_file = open(original_file_name, 'r')

    # 从文件中读取样本路径和标签值
    # >data/train/21.png )beiji
    for line in original_file:
        filename , _ , label = line[:-1].partition(' ') # partition函数只读取第一次出现的标志，分为左右两个部分,[:-1]去掉回车

        if not os.path.exists(filename):
            print("[ERROR] 图像文件不存在：%s" % filename)
            continue

        image = cv2.imread(filename)
        if image is None: continue

        _,image_name = os.path.split(filename)

        print("处理图像：",image_name)
        enhance(image,enhance_dir,image_name,new_label_file,label)

    original_file.close()
    new_label_file.close()


# 注意：需要在根目录下运行，存到 /data/train目录下
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label")
    args = parser.parse_args()


    if not args.label or not os.path.exists(args.label):
        print("目标标签文件[%s]不存在" % args.label)
        exit(-1)

    enhance_dir = "data/enhance"
    if not os.path.exists(enhance_dir):os.makedirs(enhance_dir)

    new_label_name = os.path.join(enhance_dir,"enhance.txt")
    original_file_name = args.label

    process(original_file_name,new_label_name,enhance_dir)