import re
import logging

rex = re.compile(' ')
logger = logging.getLogger("TextUitil")

# 加载字符集，charset.txt，最后一个是空格
def get_charset(charset_file):
    charset = open(charset_file, 'r', encoding='utf-8').readlines()
    charset = [ch.strip("\n") for ch in charset]
    charset = "".join(charset)
    charset = list(charset)
    return charset


# 处理一些“宽”字符
def process_unknown_charactors(sentence, dict):
    unkowns = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＠＃＄％＾＆＊（）－＿＋＝｛｝［］｜＼＜＞，．。；：､？／"
    knows = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_+={}[]|\<>,.。;:、?/"

    confuse_letters = "OolIZS"
    replace_letters = "0011zs"

    result = ""
    for one in sentence:
        # 对一些特殊字符进行替换，替换成词表的词
        i = unkowns.find(one)
        if i==-1:
            letter = one
        else:
            letter = knows[i]
            # logger.debug("字符[%s]被替换成[%s]", one, letter)

        # 看是否在字典里，如果不在，给替换成一个怪怪的字符'■'来训练，也就是不认识的字，都当做一类，这个是为了将来识别的时候，都可以明确识别出来我不认识，而且不会浪费不认识的字的样本
        # 但是，转念又一想，这样也不好，容易失去后期用形近字纠错的机会，嗯，算了，还是返回空，抛弃这样的样本把
        if letter not in dict:
            logger.error("句子[%s]的字[%s]不属于词表,剔除此样本",sentence,letter)
            #letter = '■'
            return None

        # 把容易混淆的字符和数字，替换一下
        j = confuse_letters.find(letter)
        if j!=-1:
            letter = replace_letters[j]

        result+= letter
    return result


# 将label转换为数字表示
def convert_label_to_id(label, charsets):
    # 获取label内容
    # 1.label预处理校验
    label = process_unknown_charactors(label, charsets)
    # 2.非空校验
    if label is None:
        return None
    # 3.去除空格
    label = rex.sub('', label)
    # 4.将label转为数字
    label = [charsets.index(l) for l in label]
    return label


# 按照List中最大长度扩展label
def extend_to_max_len(labels, ext_val: int = -1):
    max_len = 0
    for one in labels:
        if len(one)>max_len:
            max_len = len(one)

    for one in labels:
        one.extend( [ext_val] * (max_len - len(one)) )

    return labels


if __name__ == "__main__":
    charset = get_charset("../charset6k.txt")

    label_id = convert_label_to_id('我爱北京天安门', charset)
    for id in label_id:
        print(id, end=",")
    print("\n")

    print("=======将label数组扩展===========")
    label_id2 = convert_label_to_id('天津', charset)
    labels = []
    labels.append(label_id)
    labels.append(label_id2)
    res = extend_to_max_len(labels)
    for item in labels:
        for id in item:
            print(id, end=",")
        print("")

