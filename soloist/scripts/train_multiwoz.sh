#!/bin/bash
# lr 1e-5 to 5e-5
# mc_loss_efficient 0.1 to 1
# etc.
CUDA_VISIBLE_DEVICES=1,2 python3 -m torch.distributed.launch \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
--master_addr="localhost" \
--master_port=12026 /home/xiaoying/soloist-cuhk/soloist/soloist_train.py \
--output_dir=soloist_train5 \
--model_type=gpt2 \
--model_name_or_path=gtg_pretrained \
--do_train \
--train_data_file=/home/xiaoying/soloist-cuhk/examples/multiwoz/data/standard/train/train5.soloist.json \
--add_special_action_tokens=../examples/multiwoz/resource/special_tokens_train.txt \
--per_gpu_train_batch_size 8 \
--num_train_epochs 20 \
--learning_rate 5e-5 \
--overwrite_cache \
--save_steps 10 \
--max_seq 500 \
--overwrite_output_dir \
--max_turn 15 \
--num_candidates 1 \
--mc_loss_efficient 0.33 \
--add_response_prediction \
--add_same_belief_response_prediction \
--add_belief_prediction
