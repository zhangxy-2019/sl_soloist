#!/bin/bash
# python_file rewardmodel_evalroberta.py or rewardmodel_eval.py
# model_type roberta or bert
# etc.
CUDA_VISIBLE_DEVICES=3 python3 rewardmodel_eval.py \
--model_type=bert \
--model_name_or_path=sgd_pretrain_bertbase_epoch105e_7/checkpoint-10 \
--do_eval \
--eval_data_file=../examples/sgd_data/sgd.testsample50.json \
--per_gpu_eval_batch_size 16 \
--output_dir=sgd_pretrain_bertbase_epoch105e_7/checkpoint-10 \
--max_seq 500 \
--overwrite_output_dir \
--max_turn 15 \
--num_candidates 1 \
--add_response_prediction \
--add_same_belief_response_prediction \
--add_belief_prediction
