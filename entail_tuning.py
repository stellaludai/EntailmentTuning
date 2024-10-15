from transformers import BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch
from transformers import Trainer, TrainingArguments

import json
import torch
import os
import random

snli_path = '/path/to/your/snli_1.0_train.jsonl'
mnli_path = '/path/to/your/multinli_1.0_train.jsonl'
snli_val_path = '/path/to/your/snli_1.0_dev.jsonl'
mnli_val_path = '/path/to/your/multinli_1.0_dev_matched.jsonl'
nq_train_claim_path = '/path/to/your/nq-train-with-claim.json'
nq_dev_claim_path = '/path/to/your/nq-dev-with-claim.json'

class DataCollatorForEntailment(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.8, pad_to_multiple_of=None):
        super().__init__(tokenizer, 
                         mlm=mlm, 
                         mlm_probability=mlm_probability, 
                         pad_to_multiple_of=pad_to_multiple_of)


    def torch_call(self, examples):
        # examples: a list of input_ids
        input_ids = [e['input_ids'] for e in examples]
        attention_mask = [e['attention_mask'] for e in examples]
        premise_tokens_mask = [e['premise_tokens_mask'] for e in examples]

        batch = {
            "input_ids": _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of),
            "attention_mask": _torch_collate_batch(attention_mask, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        }
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        premise_tokens_mask = _torch_collate_batch(premise_tokens_mask, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        assert self.mlm
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], 
            special_tokens_mask=special_tokens_mask, 
            premise_tokens_mask=premise_tokens_mask
        )
 
        return batch


    def torch_mask_tokens(self, inputs, special_tokens_mask, premise_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.detach().clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        
        input_tokens_mask = special_tokens_mask | premise_tokens_mask # 如果是special token或者是premise的token，就不会被mask

        probability_matrix.masked_fill_(input_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # original bert collator: 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK]) indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

class EntailmentDataset:
    def __init__(self, nq_path, snli_path, mnli_path, tokenizer, to_predict='hypothesis'):
        self.tokenizer = tokenizer
        self.to_predict = to_predict

        self.nq_data = json.load(open(nq_path, 'r'))
        self.snli_data = self.construct_nli_data(snli_path)
        self.mnli_data = self.construct_nli_data(mnli_path)

        self.data = self.nq_data + self.snli_data + self.mnli_data
        # shuffle data
        local_random = random.Random(42)  # 创建一个局部的随机生成器
        local_random.shuffle(self.data)      # 使用局部生成器洗牌
    
    def construct_nli_data(self, nli_path):
        with open(nli_path, 'r') as file:
            # 逐行读取并解析
            data = [json.loads(line) for line in file]
        # 从snli数据中构建数据集
        nli_data = []
        for item in data:
            if item['gold_label'] == 'entailment':
                nli_data.append({
                    'positive_ctxs': [{'text': item['sentence1']}], # to align with the format of nq
                    'claim': item['sentence2'],
                })
        return nli_data
    
    def __len__(self):
        return len(self.data)
    
    def prompt(self, premise, hypothesis):
        premise_prompt = f"The passage \"{premise}\" entails that "
        hypothesis_prompt = f"{hypothesis}."
        return premise_prompt + hypothesis_prompt, premise_prompt
    
    def __getitem__(self, idx):
        claim = self.data[idx]['claim']
        positive_ctx = self.data[idx]['positive_ctxs'][0]['text'] # only one positive context
        input_text, premise_text = self.prompt(positive_ctx, claim)
        # tokenize and get premise mask
        inputs = self.tokenizer(input_text, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        premise_input_ids = self.tokenizer(premise_text, return_tensors='pt')['input_ids'].squeeze(0)
        premise_tokens_mask = torch.ones(input_ids.shape, dtype=torch.bool)
        if self.to_predict == 'hypothesis':
            premise_tokens_mask[len(premise_input_ids):] = False
        else:
            premise_tokens_mask[:len(premise_input_ids)] = False

        # labels will be get in the collator
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "premise_tokens_mask": premise_tokens_mask
        } 


def main():
    to_predict = 'hypothesis'
    mlm_probability = 0.2

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('...model and tokenizer loaded')
    # load the entailment dataset. premise, hypothesis
    train_dataset = EntailmentDataset(
        nq_path=nq_train_claim_path,
        snli_path=snli_path,
        mnli_path = mnli_path,
        tokenizer=tokenizer,
        to_predict=to_predict)
    validation_dataset = EntailmentDataset(
        nq_path=nq_dev_claim_path, 
        snli_path=snli_val_path,
        mnli_path=mnli_val_path,
        tokenizer=tokenizer,
        to_predict=to_predict)
    print('...dataset loaded')
    # collator
    data_collator = DataCollatorForEntailment(tokenizer, mlm=True, mlm_probability=mlm_probability) # 这里的mlm_probability是指hypothesis的mask概率，因为premise的不会被替换成mask

    training_args = TrainingArguments(
        output_dir=f"/home/entail_finetuned_models",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        warmup_steps=100,
        per_device_train_batch_size=32,
        num_train_epochs=10,
        eval_steps=200,
        save_total_limit=5,
        weight_decay=0.01,
        remove_unused_columns=False,
        logging_steps=10,
        report_to="wandb",
        run_name=f"entailment_finetuning_bert_hypothesis"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()


if __name__ ==  '__main__':
    main()


    