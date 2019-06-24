if [ "$1" = "" ]; then
    echo "Usage: enhance.sh <标签文件路径>"
    exit
fi

python -m data_generator.sample_enhance --label $1