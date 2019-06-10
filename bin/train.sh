if [ "$1" = "stop" ]; then
    echo "停止训练"
    kill -9 `ps aux|grep crnn| grep -v grep|awk '{print $2}'`
    exit
fi

if [ "$1" = "console" ]; then
    echo "调试模式"
    # 第一句表明使用第0个GPU，CRNN用第0个GPU，CTPN用第1个GPU，恩，我这么分配的
    CUDA_VISIBLE_DEVICES=0 \
    python \
        -m tools.train \
        --train_dir=data/ \
        --train_batch=1 \
        --train_steps=1 \
        --learning_rate=0.001 \
        --label_file=train.txt \
        --charset=charset.6883.txt \
        --name=crnn \
        --validate_steps=1 \
        --validate_file=data/test.txt \
        --validate_batch=1 \
        --early_stop=1 \
        --num_threads=4 \
        --tboard_dir=tboard \
        --debug=True
else
    echo "生产模式"
    # 第一句表明使用第0个GPU，CRNN用第0个GPU，CTPN用第1个GPU，恩，我这么分配的
    CUDA_VISIBLE_DEVICES=0 \
    python \
        -m tools.train \
        --train_dir=data/ \
        --train_batch=64 \
        --train_steps=1000000 \
        --learning_rate=0.001 \
        --label_file=train.txt \
        --charset=charset.6883.txt \
        --name=crnn \
        --validate_steps=1000 \
        --validate_file=data/test.txt \
        --validate_batch=8 \
        --early_stop=10 \
        --num_threads=4 \
        --tboard_dir=tboard \
        --debug=True \
        >> ./logs/crnn.log 2>&1 &
fi
