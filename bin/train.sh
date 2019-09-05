#!/usr/bin/env bash
Date=$(date +%Y%m%d%H%M)

if [ "$1" = "stop" ]; then
    echo "停止训练"
    kill -9 `ps aux|grep python|grep name=crnn| grep -v grep|awk '{print $2}'`
    exit
fi


if [ "$1" = "console" ]; then
    echo "调试模式"
    # 第一句表明使用第0个GPU，CRNN用第0个GPU，CTPN用第1个GPU，恩，我这么分配的
    CUDA_VISIBLE_DEVICES=0 \
    python \
        -m tools.train \
        --data_dir=data/ \
        --train_batch=3 \
        --train_steps=5 \
        --train_num_threads=1 \
        --learning_rate=0.001 \
        --label_file=data/train.txt \
        --charset=charset.3770.txt \
        --name=crnn \
        --resize_mode=PAD \
        --validate_steps=2 \
        --validate_num=2 \
        --validate_file=data/test.txt \
        --validate_batch=3 \
        --validate_num_threads=1 \
        --early_stop=2 \
        --tboard_dir=tboard \
        --debug=True
else
    echo "生产模式"
    # 第一句表明使用第0个GPU，CRNN用第0个GPU，CTPN用第1个GPU，恩，我这么分配的
    CUDA_VISIBLE_DEVICES=0 \
    nohup python \
        -m tools.train \
        --data_dir=data/ \
        --train_batch=64 \
        --train_steps=1000000000 \
        --train_num_threads=4 \
        --model=LATEST \
        --learning_rate=0.001 \
        --label_file=data/train.txt \
        --charset=charset.3770.txt \
        --name=crnn \
        --resize_mode=PAD \
        --validate_steps=1000 \
        --validate_file=data/test.txt \
        --validate_batch=32 \
        --validate_num=20 \
        --validate_num_threads=1 \
        --early_stop=100 \
        --tboard_dir=tboard \
        --debug=True \
        >> ./logs/crnn_$Date.log 2>&1 &
fi
