#!/bin/bash
# lr 5e-6
# rewardmodel_type gpt2 or bert or roberta
# positive_reward 0.5
# negative_reward -0.001
# etc.
TEMP=1
TOP_P=0.5
CUDA_VISIBLE_DEVICES=0 python3 soloistRLrewardmodel_train.py \
--output_dir=soloist-rl-hotel/soloist_hotel_rlfinetune \
--model_type=gpt2 \
--model_name_or_path=soloist-rl-hotel/soloist_hotel_augmented/checkpoint-520 \
--rewardmodel_type=gpt2 \
--rewardmodel_name_or_path=soloist-rl-hotel/soloist_hotel5_classifier/checkpoint-120 \
--do_train \
--train_data_file=../examples/multiwoz/data/standard/hotel/hotel45.valid.soloist.json \
--add_special_action_tokens=../examples/multiwoz/resource/special_tokens_hotel.txt \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--num_train_epochs 5 \
--num_samples 1 \
--learning_rate 5e-6 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--stop_token '<|endoftext|>' \
--num_candidates 1 \
--logging_steps 10 \
--overwrite_cache \
--save_steps 10 \
--max_seq 500 \
--overwrite_output_dir \
--max_turn 15 \
--positive_reward 0.5 \
--negative_reward -0.001 \
--replace_systurn
