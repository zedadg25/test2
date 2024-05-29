import wandb
import re
import nltk
import datetime
import pandas as pd
import torch
import string
import time
import numpy as np
import random
import os
import evaluate
import torch.nn as nn
import bitsandbytes as bnb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import (AdamW, AutoTokenizer,AutoModelForSeq2SeqLM,
                          T5ForConditionalGeneration, T5Tokenizer,
                          AutoConfig,
                          get_linear_schedule_with_warmup)
from transformers.optimization import Adafactor, AdafactorSchedule
from peft import LoraConfig, get_peft_model
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

class config:
  LOAD_FROM_HUB = False
  SEED = 42
  ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
  DATASET_NAME = 'readerbench/ro-text-summarization'
  
  PATH_FOR_MODELS = f'{ROOT_DIR}/models/'
  PATH_FOR_LAST_CHECKPOINT = f'{PATH_FOR_MODELS}/summary_ro-0.6169785120953605.pt'

  MODEL_MT0 = 'bigscience/mt0-large'
  MODEL_FLAN_XXL = 'google/flan-t5-xl'
  MODEL_FLAN_UL = 'google/flan-ul2'
  MODEL_MT5_RO = 'dumitrescustefan/mt5-large-romanian'
  MODEL_GPTRO = 'readerbench/RoGPT2-large'
  
  DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  WANDB_KEY = '0529a3ad694a5487302b30b185aebe0c349aa1da'

"""Model config"""
config_model = {
  "epochs": 2,
  "lr": 1.9200000000000003e-05,
  "scheduler": "decreasing",
  "checkpoint": config.MODEL_MT0,
  "batch_size": 4,
  "prompt": "Summarize in Romanian the following article:\n",
  "MAX_SOURCE_LEN": 1024,
  "MAX_TARGET_LEN": 256,
  "eps":1e-8,
  "temperature": 0.3,
  "penalty_alpha":0.6,
  "model_name_saved": "summary_ro"
}

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1
)
# https://huggingface.co/spaces/evaluate-metric/rouge
rouge_score = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def preprocess_data(text):
    text = text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    text = "Summarize in Romanian the following article:\n " + text + '\nSummary:'
    return text

def tokenize_function(text, tokenizer, max_len):
    encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
    return encoded_dict['input_ids'],encoded_dict['attention_mask']

class SumDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        text = item['Content']
        sum = item['Summary']
        return {
            'content': text,
            'summary': sum,
            'source_inputs_ids' : torch.tensor(item['source_inputs_ids']).squeeze(),
            'source_attention_mask' : torch.tensor(item['source_attention_mask']).squeeze(),
            'target_inputs_ids' : torch.tensor(item['target_inputs_ids']).squeeze()  
            }

def initialize_parameters(model, train_dataloader,optimizer_name, epochs):
    total_steps = len(train_dataloader) * epochs

    if optimizer_name=='adam':
        optimizer = AdamW(model.parameters(), lr=config_model["lr"], eps=config_model["eps"], correct_bias=False, no_deprecation_warning=True)  # noqa: E501
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)  # noqa: E501

    elif optimizer_name=='ada':
        optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True, lr=None, clip_threshold=1.0)  # noqa: E501
        scheduler = AdafactorSchedule(optimizer)

    autoconfig = AutoConfig.from_pretrained(config_model["checkpoint"])
    return optimizer, scheduler, autoconfig

def train(model, dataloader, optimizer, epoch, epochs):

    total_t0 = time.time()
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')

    train_total_loss = 0
    model.train()

    for step, batch in enumerate(dataloader):

        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        source_input_ids = batch['source_inputs_ids']
        source_attention_mask = batch['source_attention_mask']
        target_input_ids = batch['target_inputs_ids']

        optimizer.zero_grad()

        outputs = model(input_ids=source_input_ids.to(config.DEVICE, dtype=torch.long),
                        attention_mask=source_attention_mask.to(config.DEVICE, dtype=torch.long),
                        labels=target_input_ids.to(config.DEVICE, dtype=torch.long))

        loss, prediction_scores = outputs[:2]
        loss_item = loss.item()
        print('Batch loss:', loss_item)
        train_total_loss += loss_item

        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        # current_lr = optimizer.param_groups[-1]['lr']

    # calculate the average loss over all of the batches
    avg_train_loss = train_total_loss / len(dataloader)

    training_time = format_time(time.time() - total_t0)

    print("")
    print("summary results")
    print("epoch | train loss | train time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {training_time:}")
    return avg_train_loss, current_lr

"""Eval func"""

def get_rouge_score(pred, target):
  global rouge_score
  results = rouge_score.compute(predictions=pred,
                                references=target)
  # {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
  return results

def get_bleu_score(pred, target):
  global bleu
  results = bleu.compute(predictions=pred, references=target)
  results['precisions'] = np.mean(results['precisions'])
  # {'bleu': 1.0, 'precisions': [1.0, 1.0, 1.0, 1.0], 'brevity_penalty': 1.0, 'length_ratio': 1.1666666666666667, 'translation_length': 7, 'reference_length': 6}
  return {'bleu': results['bleu'], 'precisions':results['precisions'] }

def eval(model, dataloader, epoch):
    total_t0 = time.time()
    print("Running Validation...")

    model.eval()

    valid_losses = []
    rouge_scores = {}
    bleu_scores = {}

    for batch in dataloader:
        source_input_ids = batch['source_inputs_ids']
        source_attention_mask = batch['source_attention_mask']
        target_input_ids = batch['target_inputs_ids']

        with torch.no_grad():
            outputs = model(input_ids=source_input_ids.to(config.DEVICE, dtype=torch.long),
                            attention_mask=source_attention_mask.to(config.DEVICE, dtype=torch.long),
                            labels=target_input_ids.to(config.DEVICE, dtype=torch.long))

            loss, prediction_scores = outputs[:2]
            valid_losses.append(loss.item())

            generated_ids = model.generate(
                    input_ids=source_input_ids.to(config.DEVICE, dtype=torch.long),
                    attention_mask=source_attention_mask.to(config.DEVICE, dtype=torch.long),
                    temperature=config_model["temperature"],
                    use_cache=True,
                    penalty_alpha=config_model["penalty_alpha"],
                    do_sample=True,
                    no_repeat_ngram_size=1,
                    num_return_sequences=1,
                    early_stopping=True,
                    max_length=config_model["MAX_TARGET_LEN"],
                    )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]  # noqa: E501
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in target_input_ids]  # noqa: E501

            current_rouge_scores = get_rouge_score(preds, target) # {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
            current_bleu_scores = get_bleu_score(preds, target) #  {'bleu': results['bleu'], 'precisions':results['precisions'] }

            rouge_scores = {key: rouge_scores.get(key, 0) + current_rouge_scores.get(key, 0)
                            for key in set(rouge_scores) | set(current_rouge_scores)}
            bleu_scores = {key: bleu_scores.get(key, 0) + current_bleu_scores.get(key, 0)
                            for key in set(bleu_scores) | set(current_bleu_scores)}


            print('Pred:',preds,'\nTarget:',target)

    avg_val_loss = np.mean(valid_losses)

    bleu_scores = {k:v/len(dataloader) for k, v in bleu_scores.items()}
    rouge_scores = {k:v/len(dataloader) for k, v in rouge_scores.items()}

    training_time = format_time(time.time() - total_t0)

    print("")
    print("summary results")
    print("epoch | val loss | val rouge | val bleu | val time")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} |  {rouge_scores} | {bleu_scores} | {training_time:}")

    return avg_val_loss, bleu_scores, rouge_scores

""" Train loop"""
def train_loop(epochs, train_dataloader,val_dataloader, optimizer):
  global model_lora
  best_val_loss = float('inf')
  best_model=None
  run = wandb.init(project="summary_finetuning_ro_news", config=config_model)

  for epoch in range(epochs):
    train_loss, current_lr = train(model_lora, train_dataloader, optimizer, epoch, epochs)
    val_loss, val_bleu, val_rouge = eval(model_lora, val_dataloader, epoch)
    wandb.log({"Train Loss":train_loss, "Validation Loss": val_loss, "Validation Rouge": val_rouge,
               "Validation Bleu": val_bleu,"Scheduler":current_lr})  # noqa: E501
    print(f"Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Rouge: {val_rouge}, Validation Bleu: {val_bleu}, Scheduler:{current_lr}")
    print('\n')

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_model = model_lora
      best_model.save_pretrained(config.PATH_FOR_MODELS+ config_model["model_name_saved"] + f'-{best_val_loss}.pt')

  tokenizer.save_pretrained(config.PATH_FOR_MODELS+ config_model["model_name_saved"] + '-tokenizer.pt' )
  return best_model, best_val_loss


""" Load data"""
test_ds = load_dataset(config.DATASET_NAME, split="test")
train_ds = load_dataset(config.DATASET_NAME, split="train")

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_MT0)
if config.LOAD_FROM_HUB==True:
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_MT0, torch_dtype="auto", device_map="auto")
    """Check parameters"""
    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    model_lora = get_peft_model(model, peft_config)
    model_lora.print_trainable_parameters()
else:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_MT0, torch_dtype="auto", device_map="auto")
    for param in base_model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)
    base_model.gradient_checkpointing_enable()  # reduce number of stored activations
    base_model.enable_input_require_grads()
    
    model_lora = PeftModel.from_pretrained(model=base_model,model_id=config.PATH_FOR_LAST_CHECKPOINT, is_trainable=True)    
    model_lora.print_trainable_parameters()

# Fine tuning summary model
df_train = pd.DataFrame(train_ds)
df_test = pd.DataFrame(test_ds)

# df_train = df_train[:100]
# df_test = df_test[:50]

print(df_train.head())
df_train.drop(columns=['Category','Title','href','Source'], inplace=True)
df_test.drop(columns=['Category','Title','href','Source'], inplace=True)
df_train.drop_duplicates(subset=['Content'],inplace=True)
df_test.drop_duplicates(subset=['Content'],inplace=True)

df_train, df_val = train_test_split(df_train, test_size=0.10, random_state=config.SEED)

df_train['Content'] = df_train.apply(lambda x: preprocess_data(x['Content']), axis=1)
df_val['Content'] = df_val.apply(lambda x: preprocess_data(x['Content']), axis=1)
df_test['Content'] = df_test.apply(lambda x: preprocess_data(x['Content']), axis=1)

""" Tokenize"""
df_val['target_inputs_ids'],df_val['target_attention_mask']  = zip(* df_val.apply(lambda x: tokenize_function(x['Summary'],tokenizer,config_model["MAX_TARGET_LEN"]), axis=1))
df_train['target_inputs_ids'], df_train['target_attention_mask'] = zip(* df_train.apply(lambda x: tokenize_function(x['Summary'],tokenizer,config_model["MAX_TARGET_LEN"]), axis=1))

df_val['source_inputs_ids'], df_val['source_attention_mask'] = zip(* df_val.apply(lambda x: tokenize_function(x['Content'],tokenizer,config_model["MAX_SOURCE_LEN"]), axis=1))
df_train['source_inputs_ids'], df_train['source_attention_mask'] = zip(* df_train.apply(lambda x: tokenize_function(x['Content'],tokenizer,config_model["MAX_SOURCE_LEN"]), axis=1))

"""Dataset"""
train_set = SumDataset(df_train)
val_set = SumDataset(df_val)
train_dataloader = DataLoader(train_set, batch_size=config_model["batch_size"], shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_set, batch_size=config_model["batch_size"], shuffle=False, num_workers=4)


optimizer, scheduler, autoconfig = initialize_parameters(model_lora, train_dataloader, 'adam', 5)
model_lora = model_lora.to(config.DEVICE)

"""Visualize Model Parameters"""
for name, param in model_lora.named_parameters():
  print(name, param.requires_grad )


""" Run Training Loop"""
print('Train data size:', len(train_dataloader)*config_model["batch_size"])
print('Val data size:', len(train_dataloader)*config_model["batch_size"])

with torch.no_grad():
    torch.cuda.empty_cache()

best_model, best_val_loss = train_loop(config_model['epochs'], train_dataloader, val_dataloader, optimizer)

"""Test"""
df_test['target_inputs_ids'],df_test['target_attention_mask']  = zip(* df_test.apply(lambda x: tokenize_function(x['Summary'],tokenizer,config_model["MAX_TARGET_LEN"]), axis=1))
df_test['source_inputs_ids'],df_test['attention_mask']  = zip(* df_test.apply(lambda x: tokenize_function(x['Content'],tokenizer,config_model["MAX_TARGET_LEN"]), axis=1))
test_dataset = SumDataset(df_test)
test_dataloader = DataLoader(test_dataset, batch_size=config_model["batch_size"], shuffle=False)

avg_test_loss, bleu_scores_test, rouge_scores_test = eval(best_model, test_dataloader, 0)

wandb.log({"Test Loss": avg_test_loss, "Test Rouge": rouge_scores_test, "Test Bleu": bleu_scores_test})
print(f"Test Loss: {avg_test_loss}, Test Rouge: {rouge_scores_test}, Test Bleu: {bleu_scores_test}")