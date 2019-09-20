if [ "$5" = "" ]; then
    echo "Usage: imagen.sh <type:train|test|validate> <dir:data> <num:100> <worker> <charset>"
    exit
fi

python -m crnn.data_generator.crnn_generator --type=$1 --dir=$2 --num=$3 --worker=$4 --charset=$5