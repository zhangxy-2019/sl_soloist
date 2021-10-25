from __future__ import absolute_import, division, print_function, unicode_literals
import nltk
import argparse
import glob
import logging
import os
import pickle
import random
import shutil
import time
import logging
from torch._C import device
from tqdm import trange
import json,copy
import numpy as np
from collections import defaultdict, OrderedDict, Counter
import math
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
import re
RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import numpy as np
from torch.distributions import Categorical
# from reward_gtbeliefs import MultiWozEvaluatorgtbelief

import sys
sys.path.append('.')
sys.path.append('./transformers')
sys.path.append('./transformers/')

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(60)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)                              
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


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


##### begin added code for RL training for context-to-response evaluation (ground truth belief state is included) #####
class RLJsonDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', max_seq=80, max_turn=1, seperator=' & '):
        assert os.path.isfile(file_path)
        print(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, args.output_dir + '_cached_lm' + '_seqlen_' + str(max_seq) + '_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(f"Creating features from dataset file at {directory}")

            self.RLexamples = []
            self.RLtoken_ids = []
            self.generated_belief_response_ids = []
            self.examples = []
            self.token_ids = []

            system_token_id = tokenizer.convert_tokens_to_ids(['system'])
            user_token_id = tokenizer.convert_tokens_to_ids(['user'])
            induction_token_id = tokenizer.convert_tokens_to_ids(['=>'])

            examples = json.load(open(file_path))
       
            idxs = list(range(len(examples)))
            for id, example in enumerate(examples):
                history = example['history']
                context = history[-args.max_turn:]
                context_ids = []
                token_ids_for_context = []
                for cxt in context:
                    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cxt))
                    context_ids += ids
                    if 'user :' in cxt:
                        token_ids_for_context += user_token_id * len(ids)
                    else:
                        token_ids_for_context += system_token_id * len(ids)

                gt_response = example['belief'] + example['reply'] + ' ' + tokenizer.eos_token
                response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(gt_response))

                token_type_ids = token_ids_for_context + system_token_id
                context_tokens =  context_ids + induction_token_id
                assert(len(context_tokens) == len(token_type_ids))

                token_ids = token_type_ids + len(response_id) * system_token_id
                example_ids = context_tokens + response_id
                assert(len(token_ids) == len(example_ids))

                self.RLexamples.append(context_tokens)
                self.RLtoken_ids.append(token_type_ids)
                self.generated_belief_response_ids.append(response_id)
                self.examples.append(example_ids)
                self.token_ids.append(token_ids)
                    

    def __len__(self):
        return len(self.RLexamples)

    def __getitem__(self, item):
        return torch.tensor(self.RLexamples[item]), torch.tensor(self.RLtoken_ids[item]), self.generated_belief_response_ids[item], torch.tensor(self.examples[item]), torch.tensor(self.token_ids[item])

def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = RLJsonDataset(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, max_seq=args.max_seq, max_turn=args.max_turn)
    return dataset
##### end added code for RL training #####

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def evaluatesession(args, source, token_id, system_token_id):
    # evaluatesession(args, generated, token_type_id, decode_len, system_token_id)

    config_class, model_class, tokenizer_class = GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer
    config = config_class.from_pretrained(args.rewardmodel_name_or_path,
                                          cache_dir=args.rewardcache_dir if args.rewardcache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.rewardmodel_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.rewardcache_dir if args.rewardcache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.rewardmodel_name_or_path,
                                        from_tf=bool('.ckpt' in args.rewardmodel_name_or_path),
                                        config=config,
                                        cache_dir=args.rewardcache_dir if args.rewardcache_dir else None)

    if args.add_special_action_tokens:
        special_tokens = []
        for line in open(args.add_special_action_tokens):
            special_tokens.append(line.strip())
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    # preds = []
    # gt_mclabel_ids = []
    inputs = source.to(args.device)
    token_id = token_id.to(args.device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=inputs, token_type_ids=token_id)
        print("outputs: \n", len(outputs))
        # print("output[0] shape \n", outputs[0].shape)
        print("output[1] shape \n", outputs[1].shape)
        # print("output[2] shape \n", outputs[2].shape)
        logits = outputs[1].detach().cpu().numpy()
    # print("mc_logits: \n", logits)
    pred = np.argmax(logits)
    # print("pred label: \n", pred)
    score_dict = {0: args.positive_reward, 1: args.negative_reward}
    reward_score = score_dict[pred]
    # reward_score = logits[0][0]

    # print("reward_score: ", reward_score)
    return pred, reward_score

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

#### add for RL training, the input is a single generated sequence #####
def sample_batch_sequence(model, tokenizer, input_ids, token_type_ids, system_token_id, gtresponse_id, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    # model.train()
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    system_token_id = torch.tensor(system_token_id, dtype=torch.long, device=device).unsqueeze(0)
    length = len(gtresponse_id)
    gtbf_response_id = torch.tensor(gtresponse_id, dtype=torch.long, device=device)
    generated = input_ids
    context = input_ids
    log_probs = []
    past = None # using past
    # with torch.no_grad():
    for i in range(length):
        i = torch.tensor(i, dtype=torch.int, device=device)
        inputs = {'input_ids': context, 'past': past, 'token_type_ids':token_type_ids}
        outputs, past = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
        next_token_logits = outputs[:, -1, :] / (temperature if temperature > 0 else 1.)  # lm_logits # for GPT2LMHeadModel
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

        multi_dist = Categorical(F.softmax(filtered_logits, dim=-1)) # It is equivalent to the distribution that torch.multinomial() samples from.
        next_token = multi_dist.sample()
        # print("next token: \n", next_token)
        next_token = torch.tensor([gtbf_response_id[i]], dtype=torch.long, device=device)
        # print("next token in gt", next_token)
        log_prob = multi_dist.log_prob(next_token)
        next_token = next_token.unsqueeze(0) 
        log_probs.append(log_prob)
                #    next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        context = next_token
        generated = torch.cat((generated, next_token), dim=1)
        token_type_ids = system_token_id
        decode_len = len(log_probs) # length of decoded sequence
        if next_token == tokenizer.eos_token_id:
            break
    # assert decode_len == length
    log_probs = torch.stack(log_probs, dim=1)
    log_probs = torch.sum(log_probs, dim=1) / decode_len # log_probs size = torch.Size([1])  
    past = None # using past
    
    return log_probs

def train(args, train_dataset, model, tokenizer):

    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) # here we use batch_size =1
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)

    # original is AdamW
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
    model.resize_token_embeddings(len(tokenizer))
        # Train!
    logger.info("***** Running RL training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    system_token_id = tokenizer.convert_tokens_to_ids(['system'])

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # all_rewards = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for e in train_iterator:
        for step, batch in enumerate(train_dataloader):
            log_every_n_interval(500, f"  PROGRESS: {int(float(global_step)/t_total*100)}%")
            if step % 500 == 0:
                logger.info(f"  PROGRESS: {int(float(global_step)/t_total*100)}%")
            input_id, token_type_id, gtresponse_id, example_id, token_id = batch

            model.train()
            RL_log_prob = sample_batch_sequence(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_id,
                token_type_ids=token_type_id,
                system_token_id=system_token_id,
                gtresponse_id=gtresponse_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device
            )
            pred, reward = evaluatesession(args, example_id, token_id, system_token_id)
            # print("true mc_label: \n", mc_label)
            print("predicted mc_label: \n", pred)
            # all_rewards.append(reward)
            # r = (reward - np.mean(all_rewards)) / max(1e-4, np.std(all_rewards))
            loss = - reward * RL_log_prob
            print("reward: \n", reward)
            print("log prob: \n", RL_log_prob)
            print("loss: \n", loss)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                # print("loss / args.gradient_accumulation_steps: \n", loss)
            tr_loss += loss.item() # float
            # print("tr_loss: \n", tr_loss)
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
                    # print("logging_loss: \n", logging_loss)
                    logger.info(f" EVALERR:  {(tr_loss - logging_loss)/float(args.logging_steps)}")

                    logging_loss = copy.deepcopy(tr_loss) # float
                    
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                        # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_RLexamples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)

        inputs, tokens, labels, masks = batch
            # import pdb
            # pdb.set_trace()
        inputs = inputs.to(args.device)
        tokens = tokens.to(args.device)
        labels = labels.to(args.device)
        masks = masks.to(args.device)
        # inputs = inputs.to(args.device)
        # labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels, token_type_ids=tokens) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def main():
    parser = argparse.ArgumentParser()
        ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
                            ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_type", default='gpt2', type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--rewardmodel_type", default='bert', type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--rewardcache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=80, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to train on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the eval set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--rewardmodel_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
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
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
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

    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--add_special_action_tokens", default='', type=str)
    parser.add_argument("--length", type=int, default=110)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument('--stop_token', type=str, default='<|endoftext|>', help="Token at which text generation is stopped")
    parser.add_argument('--max_turn', type=int, default=15, help="number of turns used as context")
    parser.add_argument("--max_seq", default=200, type=int, help="")
    parser.add_argument("--train_mode", default='RLtrain', type=str, help="RLvalid/RLtrain")
    parser.add_argument("--num_candidates", default=1, type=int, help="")
    parser.add_argument("--positive_reward", default=1.0, type=float)
    parser.add_argument("--negative_reward", default=-0.001, type=float)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)


    args = parser.parse_args()
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args)

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    # config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
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

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    # args.rewardmodel_type = args.rewardmodel_type.lower()
    logger.info("RL Training parameters %s", args)
     # Training
    if args.do_train:

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and args.local_rank == -1:
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
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
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

if __name__ == '__main__':
    main()