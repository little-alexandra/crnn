# 用来查看某个词对应的标注和图片情况，用来验证识别效果用的
label_file="data/test.txt"
keyword="财衬通支"
grep $keyword $label_file |awk {'print $1'}| xargs -I {} see {}