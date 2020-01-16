for ((i=0;i<5;i++));
do
  python main.py \
  --do_train \
  --data_dir data/nlp_new_train$i\
  --out_dir Five_Limodel_bert/model_$i\
  --resume_model_path Five_Limodel_bert/model_$i/pytorch_model.bin
done