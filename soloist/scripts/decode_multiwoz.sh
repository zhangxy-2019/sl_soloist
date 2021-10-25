#!/bin/bash
# temp 0.7 - 1.5
# top_p 0.2 - 0.8
# CHECKPOINT saved checkpints, valid around 40k to 80k
NS=1
TEMP=1
TOP_P=0.5
CHECKPOINT=/home/xiaoying/soloist-cuhk/soloist/soloist_train5_mt/checkpoint-140
CUDA_VISIBLE_DEVICES=0 python3 soloist_decode_checkpoints.py \
--model_type=gpt2 \
--model_name_or_path=${CHECKPOINT} \
--num_samples ${NS} \
--input_file=/home/xiaoying/soloist-cuhk/examples/multiwoz/data/standard/train/train198.test.soloist.json \
--top_p ${TOP_P} \
--temperature ${TEMP} \
--max_turn 15 