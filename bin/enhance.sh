echo "=================="
echo "=  图片增强"
echo "=================="

if [ "$2" = "" ]; then
    echo "Usage: enhance.sh <标签文件路径> <字符集文件路径>"
    exit
fi

python -m data_generator.sample_enhance --label $1 --charset $2