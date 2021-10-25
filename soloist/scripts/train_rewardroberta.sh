#!/bin/bash
# lr 5e-8 to 5e-7
# batch_size 16
# model_name_or_path roberta-base or roberta-large
# etc.
CUDA_VISIBLE_DEVICES=1 python3 rewardmodel_train_roberta.py \
--output_dir=sgd_pretrain_roberta_base_epoch10lr5e_8 \
--model_type=roberta \
--model_name_or_path=roberta-base \
--do_train \
--train_data_file=../examples/sgd_data/sgd.train.devsample50.json \
--per_gpu_train_batch_size 16 \
--num_train_epochs 10 \
--learning_rate 5e-8 \
--save_steps 2500 \
--max_seq 500 \
--overwrite_output_dir \
--max_turn 15 \
--num_candidates 1 \
--add_response_prediction \
--add_same_belief_response_prediction \
--add_belief_prediction