#!/bin/bash
# lr 5e-6
# rewardmodel_type gpt2
# positive_reward 0.5
# negative_reward -0.001
# etc.
TEMP=1
TOP_P=0.5
CUDA_VISIBLE_DEVICES=0 python3 soloistRLrewardmodel_human_in_the_loop.py \
--output_dir=/home/xiaoying/soloist-cuhk/soloist/soloist_rl_restaurant/soloist_restaurant_hm \
--model_type=gpt2 \
--model_name_or_path=/home/xiaoying/soloist-cuhk/soloist/soloist_rl_restaurant/soloist_restaurant_augmented/checkpoint-410 \
--rewardmodel_type=gpt2 \
--rewardmodel_name_or_path=/home/xiaoying/soloist-cuhk/soloist/soloist_rl_restaurant/soloist_restaurant_save5epoch20/checkpoint-120 \
--do_train \
--train_data_file=../examples/multiwoz/data/standard/restaurant/human_bot_interaction_logs.json \
--add_special_action_tokens=../examples/multiwoz/resource/special_tokens_restaurant.txt \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--num_train_epochs 5 \
--num_samples 1 \
--learning_rate 5e-6 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--stop_token '<|endoftext|>' \
--logging_steps 5 \
--overwrite_cache \
--save_steps 5 \
--max_seq 500 \
--overwrite_output_dir \
--max_turn 15 \
--positive_reward 1.0 \
--negative_reward -0.001
