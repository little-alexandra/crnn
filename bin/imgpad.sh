if [ "$1" = "" ]; then
    echo "Usage: imgpad.sh <type:train|test|validate>"
    exit
fi

python -m crnn.data_generator.imgpad --type=$1