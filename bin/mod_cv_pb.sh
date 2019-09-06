#!/usr/bin/env bash

echo "模型转换 ckpt to pb"

ckpt_mod_path=$1
save_mod_dir=$2

if [ "ckpt_mod_path" = "" ]; then
    echo "ckpt模型目录不能为空 参数1"
    exit
fi

if [ "$save_mod_dir" = "" ]; then
    save_mod_dir="model/crnn"
fi

python -m utils.mod_cv_pb.convert_model_1 \
    --ckpt_mod_path=$ckpt_mod_path \
    --save_mod_dir=$save_mod_dir \
    --charset=charset.3770.txt