#!/usr/bin/env python
HIDDEN_UNITS = 256
HIDDEN_LAYERS = 2
INPUT_SIZE = (32, 256) # 宽都统一了？，对，我看了几个别的crnn的项目代码，都是这样做的
BEAM_WIDTH = 1
WIDTH_REDUCE_TIMES = 4
SEQ_LENGTH = int(INPUT_SIZE[1] / WIDTH_REDUCE_TIMES)

RESIZE_MODE_PAD="PAD"
RESIZE_MODE_FIX="FIX"

# 定义字符集路径，相对于crnn/utils/data_utils.py的路径，所以是../config/
CHARSET_FILE='../config/charset.3770.txt'
