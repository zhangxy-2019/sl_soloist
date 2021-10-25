# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import copy
import pickle
import re
import shutil
import time
import json
import sys
sys.path.append('.')
sys.path.append('./transformers')
sys.path.append('./transformers/')

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  BertConfig,
                                  BertForSequenceClassification, BertTokenizer
                                )

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import glue_compute_metrics as compute_metrics

logger = logging.getLogger(__name__)

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
#                                                                                 RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

s = 0
s_time = time.time()
def log_every_n_interval(interval, msg):
    global s, s_time
    s += 1
    if s % interval == 0:
        if s_time is None:
            s_time = time.time()
            iter_p_s = 0
        else:
            e_time = time.time()
            elapse = e_time - s_time
            iter_p_s = interval / elapse
            s_time = e_time
        logger.info(f'MSG: {msg};  ITER: {iter_p_s:.2f}/s')


class JsonDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', max_seq=80, max_turn=1, seperator=' & '):
        # assert os.path.isfile(file_path)
        # directory, filename = os.path.split(file_path)
        # cached_features_file = os.path.join(directory, args.output_dir + '_cached_lm' + '_seqlen_' + str(max_seq) + '_' + filename)

        # if os.path.exists(cached_features_file) and not args.overwrite_cache:
        #     logger.info("Loading features from cached file %s", cached_features_file)
        #     with open(cached_features_file, 'rb') as handle:
        #         self.examples = pickle.load(handle)
        # else:
        #     logger.info(f"Creating features from dataset file at {directory}")

        self.examples = []
        self.token_ids = []
        self.attention_masks = []

        # self.mc_token_ids = []
        self.labels = []

        system_token_id = tokenizer.convert_tokens_to_ids(['system'])
        user_token_id = tokenizer.convert_tokens_to_ids(['user'])
        induction_token_id = tokenizer.convert_tokens_to_ids(['=>'])

        examples = json.load(open(file_path))
        print("length of examples: \n", len(examples))
        response_pool = []
        belief_pool = []
        idxs = list(range(len(examples)))
        for i in examples:
            response = i['reply'] + ' '+ tokenizer.sep_token
            response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response))
            response_pool.append(response_id)

            belief_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i['belief']))
            belief_pool.append(belief_id)

        for id, example in enumerate(examples):
            history = example['history']
            context = history[-max_turn:]
            context_ids = []
            token_ids_for_context = []
            for cxt in context:
                ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cxt))
                context_ids += ids
                if 'user :' in cxt:
                    token_ids_for_context += [0] * len(ids)
                else:
                    token_ids_for_context += [0] * len(ids)

            history = ' '.join(history[-max_turn:])
            # kb = ' '#example['kb']
            belief = example['belief']
            
            belief_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(belief))
            response =  example['reply'] + ' ' + tokenizer.sep_token
            response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response))

            token_id = [0] + token_ids_for_context + [0] + [1] * len(belief_id) + [1] * len(response_id)
            source = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id] + belief_id + response_id
            print("source: \n", source)
            if len(source) < max_seq:
                attention_mask = [0] * max_seq
                attention_mask[:len(source)] = [1] * len(source)
                # self.mc_token_ids.append(len(source) - 1)
                source += [0] * (max_seq - len(source))
                token_id += [0] * (max_seq - len(token_id))
            else:
                attention_mask = [1] * max_seq
                # self.mc_token_ids.append(max_seq - 1)
                source = source[-max_seq:]
                token_id = token_id[-max_seq:]

            self.labels.append(0)
            
            if not  len(source) == len(attention_mask):
                import pdb
                pdb.set_trace()

            self.examples.append(source)
            self.token_ids.append(token_id)
            self.attention_masks.append(attention_mask)

            if args.add_same_belief_response_prediction:
                for _ in range(args.num_candidates):

                    sample_idxs = copy.deepcopy(idxs)
                    sample_idxs.remove(id)
                    random_idx = random.choice(sample_idxs)
                    # random_idx = random.choice(idxs)
                    new_response_id = response_pool[random_idx]
                    new_belief_id = belief_pool[random_idx]

                    source = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id] + new_belief_id + new_response_id
                    token_id = [0] + token_ids_for_context + [0] + [1] * len(new_belief_id)  + [1] * len(new_response_id)
                    if len(source) < max_seq:
                        attention_mask = [0] * max_seq
                        attention_mask[:len(source)] = [1] * len(source)
                        # self.mc_token_ids.append(len(source)-1)
                        source += [0] * (max_seq - len(source))
                        token_id += [0] * (max_seq - len(token_id))
                    else:
                        attention_mask = [1] * max_seq
                        # self.mc_token_ids.append(max_seq - 1)
                        source = source[-max_seq:]
                        token_id = token_id[-max_seq:]


                    self.examples.append(source)
                    self.token_ids.append(token_id)
                    self.labels.append(1)
                    self.attention_masks.append(attention_mask)
            
            if args.add_response_prediction:
                for _ in range(args.num_candidates):

                    sample_idxs = copy.deepcopy(idxs)
                    sample_idxs.remove(id)
                    random_idx = random.choice(sample_idxs)
                    # random_idx = random.choice(idxs)
                    new_response_id = response_pool[random_idx]
                    new_belief_id = belief_pool[random_idx]

                    source = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id] + belief_id + new_response_id
                    token_id = [0] + token_ids_for_context + [0] + [1] * len(belief_id)  + [1] * len(new_response_id)
                    if len(source) < max_seq:
                        attention_mask = [0] * max_seq
                        attention_mask[:len(source)] = [1] * len(source)
                        # self.mc_token_ids.append(len(source)-1)
                        source += [0] * (max_seq - len(source))
                        token_id += [0] * (max_seq - len(token_id))
                    else:
                        attention_mask = [1] * max_seq
                        # self.mc_token_ids.append(max_seq - 1)
                        source = source[-max_seq:]
                        token_id = token_id[-max_seq:]


                    self.examples.append(source)
                    self.token_ids.append(token_id)
                    self.labels.append(1)
                    self.attention_masks.append(attention_mask)

            if args.add_belief_prediction:
                for _ in range(args.num_candidates):
                    
                    sample_idxs = copy.deepcopy(idxs)
                    sample_idxs.remove(id)
                    random_idx = random.choice(sample_idxs)
                    # random_idx = random.choice(idxs)
                    new_response_id = response_pool[random_idx]
                    new_belief_id = belief_pool[random_idx]

                    source = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id] + new_belief_id + response_id
                    token_id = [0] + token_ids_for_context + [0] + [1] * len(new_belief_id)  + [1] * len(response_id)
                    if len(source) < max_seq:
                        attention_mask = [0] * max_seq
                        attention_mask[:len(source)] = [1] * len(source)
                        # self.mc_token_ids.append(len(source)-1)
                        source += [0] * (max_seq - len(source))
                        token_id += [0] * (max_seq - len(token_id))
                    else:
                        attention_mask = [1] * max_seq
                        # self.mc_token_ids.append(max_seq - 1)
                        source = source[-max_seq:]
                        token_id = token_id[-max_seq:]


                    self.examples.append(source)
                    self.token_ids.append(token_id)
                    self.labels.append(1)
                    self.attention_masks.append(attention_mask)

            if args.cut_half_response:
                for _ in range(args.num_candidates):
                    new_response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example['reply']))
                    print("new_response_id: \n", new_response_id)
                    cut_id = int(len(new_response_id)/2)
                    new_response_id = new_response_id[:cut_id]
                    print("new_response_id after: \n", new_response_id)
                    response_end = ' '+ tokenizer.sep_token
                    response_end_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_end))
                    new_response_id += response_end_id
                    # new_belief_id = belief_pool[random_idx]

                    source = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id] + belief_id + new_response_id
                    token_id = [0] + token_ids_for_context + [0] + [1] * len(belief_id)  + [1] * len(new_response_id)
                    if len(source) < max_seq:
                        attention_mask = [0] * max_seq
                        attention_mask[:len(source)] = [1] * len(source)
                        # self.mc_token_ids.append(len(source)-1)
                        source += [0] * (max_seq - len(source))
                        token_id += [0] * (max_seq - len(token_id))
                    else:
                        attention_mask = [1] * max_seq
                        # self.mc_token_ids.append(max_seq - 1)
                        source = source[-max_seq:]
                        token_id = token_id[-max_seq:]


                    self.examples.append(source)
                    self.token_ids.append(token_id)
                    self.labels.append(1)
                    self.attention_masks.append(attention_mask)

            if args.repeat_tokens:
                for _ in range(args.num_candidates):
                    new_response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example['reply']))
                    print("new_response_id: \n", new_response_id)
                    sample_idxs = copy.deepcopy(list(range(len(new_response_id))))
                    random_idx = random.choice(sample_idxs)
                    insert_token_id = new_response_id[random_idx]
                    new_response_id.insert(random_idx, insert_token_id)
                    new_response_id.insert(random_idx, insert_token_id)
                    new_response_id.insert(random_idx, insert_token_id)
                    print("new_response_id after: \n", new_response_id)
                    response_end = ' '+ tokenizer.sep_token
                    response_end_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_end))
                    new_response_id += response_end_id
                    # new_belief_id = belief_pool[random_idx]

                    source = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id] + belief_id + new_response_id
                    token_id = [0] + token_ids_for_context + [0] + [1] * len(belief_id)  + [1] * len(new_response_id)
                    if len(source) < max_seq:
                        attention_mask = [0] * max_seq
                        attention_mask[:len(source)] = [1] * len(source)
                        # self.mc_token_ids.append(len(source)-1)
                        source += [0] * (max_seq - len(source))
                        token_id += [0] * (max_seq - len(token_id))
                    else:
                        attention_mask = [1] * max_seq
                        # self.mc_token_ids.append(max_seq - 1)
                        source = source[-max_seq:]
                        token_id = token_id[-max_seq:]


                    self.examples.append(source)
                    self.token_ids.append(token_id)
                    self.labels.append(1)
                    self.attention_masks.append(attention_mask)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.attention_masks[item]), torch.tensor(self.labels[item]), torch.tensor(self.token_ids[item])

def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = JsonDataset(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, max_seq=args.max_seq, max_turn=args.max_turn)
    return dataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(train_dataloader):

            log_every_n_interval(500, f"  PROGRESS: {int(float(global_step)/t_total*100)}%")
            if step % 500 == 0:
                logger.info(f"  PROGRESS: {int(float(global_step)/t_total*100)}%")

            inputs, masks, labels, tokens = batch

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            masks = masks.to(args.device)
            tokens = tokens.to(args.device)
            model.train()
            inputs = {'input_ids': inputs,
                      'attention_mask': masks,
                      'labels': labels,
                      'token_type_ids': tokens}
            # if args.model_type in ['bert', 'xlnet'] :
            #     inputs['token_type_ids'] = tokens 
            # else:
            #     None 
            # if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            # if args.max_steps > 0 and global_step > args.max_steps:
            #     epoch_iterator.close()
            #     break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--max_seq", default=200, type=int, help="")
    parser.add_argument("--max_turn", default=1, type=int, help="")
    parser.add_argument("--num_candidates", default=1, type=int, help="")
    parser.add_argument("--add_special_action_tokens", default='', type=str)
    parser.add_argument("--add_same_belief_response_prediction", action='store_true')
    parser.add_argument("--add_response_prediction", action='store_true')
    parser.add_argument("--add_belief_prediction", action='store_true')
    parser.add_argument("--cut_half_response", action='store_true')
    parser.add_argument("--repeat_tokens", action='store_true')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)


    if args.add_special_action_tokens:
        special_tokens = []
        for line in open(args.add_special_action_tokens):
            special_tokens.append(line.strip())
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
