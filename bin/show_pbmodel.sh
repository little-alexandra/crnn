if [ "$1" = "" ]; then
    echo "模型目录不能为空: show_pbmodel <模型目录> <tboard log目录>"
    exit
fi

if [ "$2" = "" ]; then
    echo "tensorboard log目录不能为空: show_pbmodel <模型目录> <tboard log目录>"
    exit
fi

python -m utils.import_pb_to_tensorboard.py --model_dir $1 --log_dir $2