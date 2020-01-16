for ((i=0;i<5;i++));
do
  python adv_main.py \
  --do_train \
  --do_eval \
  --test \
  --data_dir data/nlp_new_train$i\
  --out_dir Five_adv_Limodel_bert_base/model_$i\
  --resume_model_path Five_adv_Limodel_bert_base/model_$i/pytorch_model.bin
done