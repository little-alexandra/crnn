echo "帮助：\n\t如果不指定文件名，识别data/test/目录下的所有图片，否则具体的照片"
echo "\t如果指定model名字，就加载，否则，加载最新的模型名字"
python -m tools.pred \
    --crnn_model_dir=model \
    --dir=data/validate \
    --charset=charset.6883.txt \
    --label=data/validate.txt\
    --debug=False \
    --crnn_model_file=crnn_2019-06-12-11-07-43.ckpt-100000
