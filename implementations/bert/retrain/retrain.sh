#!/bin/bash

# python3 BERT4doc-Classification/codes/further-pre-training/create_pretraining_data.py \
#  --input_file=Twitter_corpus-not-processed.txt \
#  --output_file=tf_twitter-not-processed.tfrecord \
# --vocab_file=../wwm_uncased_L-24_H-1024_A-16/vocab.txt \
# --do_lower_case=True \
# --max_seq_length=64 \
# --max_predictions_per_seq=20 \
# --masked_lm_prob=0.15 \
# --random_seed=12345 \
# --dupe_factor=5


# python3 BERT4doc-Classification/codes/further-pre-training/run_pretraining.py \
#  --input_file=tf_twitter-not-processed.tfrecord \
#  --output_dir=wwm_uncased_L-24_H-1024_A-16_twitter_pretrain-not-processed \
#  --do_train=True \
#  --do_eval=True \
#  --bert_config_file=../wwm_uncased_L-24_H-1024_A-16/bert_config.json \
#  --init_checkpoint=../wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
#  --train_batch_size=64 \
#  --eval_batch_size=64 \
#  --max_seq_length=64 \
#  --max_predictions_per_seq=20 \
#  --num_train_steps=100000 \
#  --num_warmup_steps=10000 \
#  --save_checkpoints_steps=10000 \
#  --learning_rate=5e-5


python3 BERT4doc-Classification/codes/fine-tuning/convert_tf_checkpoint_to_pytorch.py \
   --tf_checkpoint_path ./wwm_uncased_L-24_H-1024_A-16_twitter_pretrain-not-processed/model.ckpt-90000 \
   --bert_config_file ../wwm_uncased_L-24_H-1024_A-16//bert_config.json \
   --pytorch_dump_path  ./wwm_uncased_L-24_H-1024_A-16_twitter_pretrain-not-processed/pytorch_model.bin
