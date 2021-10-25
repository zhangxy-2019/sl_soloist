#!/bin/bash
# lr 1e-5 to 5e-5
# mc_loss_efficient 0.1 to 1
# etc.
CUDA_VISIBLE_DEVICES=4 python3 soloist_train.py \
--output_dir=pretrained_models \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_train \
--train_data_file=/data/xyzhang/RL_research_folder/micro_research/soloist-cuhk/soloist/data/sgd.h10.json \
--per_gpu_train_batch_size 16 \
--num_train_epochs 10 \
--learning_rate 5e-5 \
--overwrite_cache \
--save_steps 5000 \
--max_seq 500 \
--overwrite_output_dir \
--max_turn 15 \
--num_candidates 1 \
--mc_loss_efficient 0.33 \
--add_response_prediction \
--add_belief_prediction \
--add_same_belief_response_prediction