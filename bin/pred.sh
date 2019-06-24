if [ "$1" == "--help" ]; then
    echo "pred.sh                                   预测data/validate目录下，验证文件为data/validate.txt"
    echo "pred.sh --image xx/yy.png                 预测xx/yy.png的结果"
    echo "pred.sh --dir xxx --label yyy             批量检测并且计算正确率"
    echo "        --beam 3 --model xxx"
    echo "        --charset zzz"
    exit
fi

# 默认值
DIR="data/test"
LABEL="data/test.txt"
CHARSET="charset.6883.txt"
MODEL="LATEST" #自动加载最新的
#MODEL="crnn_2019-06-12-11-07-43.ckpt-100000"    #指定模型，也可以从外面指定
BEAM=1

ARGS=`getopt -o i:d:l:b --long image:,dir:,label:,beam:,charset:,model: -- "$@"`
eval set -- "${ARGS}"
while true ;
do
        case "$1" in
                --image)
                    echo "解析参数，图片：$2"
                    IMAGE=$2
                    shift 2
                    ;;
                --dir)
                    echo "解析参数，目录：$2"
                    DIR=$2
                    shift 2
                    ;;
                --label)
                    echo "解析参数，标签：$2"
                    LABEL=$2
                    shift 2
                    ;;
                --beam)
                    echo "解析参数，Beam：$2"
                    BEAM=$2
                    shift 2
                    ;;
                --model)
                    echo "解析参数，模型：$2"
                    MODEL=$2
                    shift 2
                    ;;
                --charset)
                    echo "解析参数，字符集：$2"
                    CHARSET=$2
                    shift 2
                    ;;
                --) shift ; break ;;
                *) echo "解析参数错误"; exit 1 ;;
        esac
done

echo "识别目录 : $DIR,\n识别标签 : $LABEL,\nBeam     : $BEAM,\n字符集   : $CHARSET,\n模型     ：$MODEL"

if [ -n "$IMAGE" ]; then
    echo "单独识别图片：$IMAGE"
    python -m tools.pred \
    --crnn_model_dir=model \
    --crnn_model_file=$MODEL \
    --file=$IMAGE \
    --charset=$CHARSET \
    --debug=True \
    --label=$LABEL \
    --beam_width=$BEAM
    see $IMAGE
    exit
fi



echo "开始批量识别..."
python -m tools.pred \
    --crnn_model_dir=model \
    --crnn_model_file=$MODEL \
    --dir=$DIR \
    --charset=$CHARSET \
    --debug=True \
    --label=$LABEL \
    --beam_width=$BEAM