if [ "$1" == "--help" ]; then
    echo "pred.sh                                   预测data/validate目录下，验证文件为data/validate.txt"
    echo "pred.sh --image xx/yy.png                 预测xx/yy.png的结果"
    echo "pred.sh --dir xxx --label yyy --beam 3    批量检测并且计算正确率"
    exit
fi

DIR="data/validate"
LABEL="data/validate.txt"
BEAM=1

ARGS=`getopt -o i:d:l:b --long image:,dir:,label:,beam: -- "$@"`
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
                --) shift ; break ;;
                *) echo "解析参数错误"; exit 1 ;;
        esac
done

if [ -n "$IMAGE" ]; then
    echo "单独识别图片：$IMAGE, BEAM: $BEAM"
    python -m tools.pred \
    --crnn_model_dir=model \
    --crnn_model_file=crnn_2019-06-12-11-07-43.ckpt-100000 \
    --file=$IMAGE \
    --charset=charset.6883.txt \
    --debug=True \
    --beam_width=$BEAM
    see $IMAGE
    exit
fi


echo "识别目录 : $DIR, 识别标签 : $LABEL, Beam Width : $BEAM"
echo "开始批量识别..."
python -m tools.pred \
    --crnn_model_dir=model \
    --crnn_model_file=crnn_2019-06-12-11-07-43.ckpt-100000 \
    --dir=$DIR \
    --charset=charset.6883.txt \
    --debug=True \
    --label=$LABEL \
    --beam_width=$BEAM