#!/bin/bash
# lr 5e-7 to 5e-6
# batch_size 16
# model_name_or_path bert-base-uncased or bert-large-uncased
# etc.
CUDA_VISIBLE_DEVICES=0 python3 rewardmodel_train.py \
--output_dir=sgd_pretrain_bertbase_epoch105e_7 \
--model_type=bert \
--model_name_or_path=bert-base-uncased \
--do_train \
--train_data_file=../examples/sgd_data/sgd.train.devsample50.json \
--per_gpu_train_batch_size 16 \
--num_train_epochs 10 \
--learning_rate 5e-7 \
--save_steps 5 \
--max_seq 500 \
--overwrite_output_dir \
--max_turn 15 \
--num_candidates 1 \
--add_response_prediction \
--add_same_belief_response_prediction \
--add_belief_prediction