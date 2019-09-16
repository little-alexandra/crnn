# 这个命令是，为了把pb model的结构输出到summary中，
# 以便可以通过tensorboard去查看
if [ "$1" = "" ]; then
    echo "模型目录不能为空: show_pbmodel <模型目录> <tboard log目录>"
    exit
fi

if [ "$2" = "" ]; then
    echo "tensorboard log目录不能为空: show_pbmodel <模型目录> <tboard log目录>"
    exit
fi

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m utils.import_pb_to_tensorboard --model_dir $1 --log_dir $2