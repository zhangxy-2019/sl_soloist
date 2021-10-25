# SL-Agent

## Installation
Require python 3.6.  
The interactive interface requries *vue* framework. Please refer to [here](https://cn.vuejs.org/index.html) for installation.

Please use the below commands to clone the repo and install required package.
```
git clone 
pip install -r requirements.txt
```
Fetch and unzip the pretrained model based on which to continue finetune your own data. (*will release more versions of pretrained models, stay tuned*) 

```zsh
wget https://bapengstorage.blob.core.windows.net/soloist/gtg_pretrained.tar.gz
cd soloist
tar -xvf gtg_pretrained_model.tar.gz
```
## SOLOIST Pipeline
**Data format**
```json
  {
    "history": [
      "user : later I have to call Jessie"
    ],
    "belief": "belief : name = Jessie",
    "kb": "",
    "reply": "system : Sure, added to your reminder list. FOLLOWUP action_set_reminder"
  },
```
We use json to represent a training example. As shown in the example, it contains the following fields:
* **history** - The context from session beginning to current turn
* **belief** - The belief state. 
* **kb** - Database query results, leave it as blank if db query is not requried.
* **reply** - The target system respose. It can be a template, an api call or natural language.

**Training SOLOIST**
```python
# under soloist folder
python train.py --output_dir=MODEL_SAVE_PATH --model_type=gpt2 --model_name_or_path=PRE_TRAINED_MODEL_PATH --do_train --train_data_file=TRAIN_FILE  --per_gpu_train_batch_size 4 --num_train_epochs EPOCH --learning_rate 5e-5 --overwrite_cache --use_tokenize --save_steps 10000 --max_seq 500 --overwrite_output_dir --max_turn 15 --num_candidates 1 --mc_loss_efficient 0.2 --add_special_action_tokens --with_code_loss --add_belief_prediction --add_response_prediction --add_same_belief_response_prediction
```
<code>output_dir </code>: Path of the saving model.  
<code>model_name_or_path </code>: Initial checkpoint;  
<code>num_train_epochs </code>: Number of training epochs;  5, 10, 20 are enough for a reasonable performance.  
<code>learning_rate </code>: Learning rate; 5e-5, 1e-5, or 1e-4.  
<code>num_candidates </code>: number of candidate; recommend 1.  
<code>mc_loss_efficient </code>: contrastive loss coefficient; 0.1 to 1.  
<code>add_belief_prediction </code>: if add contrastive loss item for belief span.  
<code>add_response_prediction </code>: if add contrastive loss item for response prediction.  
<code>add_same_belief_response_prediction </code>: if add contrastive loss item for belief span and response.  

**Generation (Decoding)**
```python
# under soloist folder
python generate.py --model_type=gpt2 --model_name_or_path=SAVED_MODEL --num_samples NS --input_file=TEST_FILE --top_p TOP_P --temperature TEMP --output_file=OUTPUT_FILE --max_turn 15
```
<code>model_name_or_path </code>: Path of the saved model.  
<code>num_samples </code>: Number of samples; 1 or 5 for reranking.  
<code>top_p </code>: Nuclear sampling; 0.2 - 0.5  
<code>temperature </code>: Nuclear sampling; 0.7 - 1.5  
<code>input_file </code>: Path to input file.  
<code>output_file </code>: Path to save results.

## SL-SOLOIST
**Classifier Training**
```bash
# under soloist folder (dialog model/classifier)
sh scripts/train_multiwoz.sh
```

```bash
# under soloist folder (roberta classifier)
sh scripts/train_rewardroberta.sh
```

```bash
# under soloist folder (bert classifier)
sh scripts/train_reward_bert.sh
```
**RL Refining**
```python
# under soloist folder
python RLrewardmodel_train.py --output_dir=MODEL_SAVE_PATH --model_type=gpt2 --model_name_or_path=TRAINED_MODEL_PATH --rewardmodel_type=gpt2 --rewardmodel_name_or_path=TRAINED_CLASSIFIER_PATH --do_train --train_data_file=TRAIN_FILE  --add_special_action_tokens=TOKEN_FILE --per_gpu_train_batch_size 1 --num_train_epochs EPOCH --learning_rate 5e-6 
--num_samples 1 --max_grad_norm 1.0 --adam_epsilon 1e-8  --stop_token '<|endoftext|>' --save_steps 10 --max_seq 500 --overwrite_output_dir --max_turn 15 --num_candidates 1 --logging_steps 10 --overwrite_cache --max_turn 15  --positive_reward 0.5 --negative_reward -0.001 --replace_systurn
```
<code>output_dir </code>: Path of the saving model.  
<code>model_name_or_path </code>: Initial dialog model checkpoint.
<code>rewardmodel_name_or_path </code>: Classifier model type ; gpt2, bert, or roberta.
<code>rewardmodel_name_or_path </code>: Classifier checkpoint.
<code>num_train_epochs </code>: Number of training epochs; 5 is enough.  
<code>train_data_file </code>: Path of interactive log data; in simulated setting using valid data. 
<code>learning_rate </code>: Learning rate; 5e-6.  
<code>num_candidates </code>: number of candidate; recommend 1.  
<code>positive_reward </code>: positive reward score ; 0.5 to 1.
<code>negative_reward </code>: negative reward score ; -0.001.  
<code>replace_systurn </code>: if add simulated interactive data contains negative example with replaced response.  

```bash
# under soloist folder (in simulated setting)
sh scripts/RLsoloistrewardmodel_negative1_fixbelief.sh
```

```bash
# under soloist folder (in real log setting)
sh scripts/RLsoloistrewardmodel_calculatebelief.sh
```

**Decoding Checkpoints**
```bash
# under soloist folder (in real log setting)
sh scripts/decode_multiwoz.sh
```

## Pre-training with homogeneous dialog corpora
Theoretically, all dialog related corpora or domain-specific corpous, like insurrance, banking domian dialog log sesssions, can be used to pre-train soloist. For simplicity, in this repo we showcase pretraining with publicly available corpus. Incorporating other dataset is straightforward.

```bash
sh scripts/pretrain_preprocessing.sh
```
**Training**

```bash
# Pretrained soloist (dialog model/classifier)
# under soloist folder
sh scripts/soloist_pretrain.sh
```

```bash
# Pretrained bert classifier
# under soloist folder
sh scripts/train_reward_bert.sh
```

```bash
# Pretrained roberta classifier
# under soloist folder
sh scripts/train_rewardroberta.sh
```
**Evaluating**

```bash
# Evaluate pretrained bert classifier using rewardmodel_eval.py
# evaluate roberta classifier using rewardmodel_evalroberta.py
# output_dir = path to multiple checkpoints / single checkpoint
# under soloist folder
sh scripts/eval_reward.sh
```
With 8 v100 GPUs, checkpoints at step 300k - 500k are able to demonstrate good transfer learning capability.

# sl_soloist
# sl_soloist
# sl_soloist
