# src/fine_tune_llama2_lora.py
# Script to fine-tune LLaMA-2-7b-hf with LoRA adapters using HuggingFace Trainer and BitsAndBytes 4-bit quantization

import os
# Disable BitsAndBytes environment check to allow quantization of sharded PyTorch weights
os.environ["TRANSFORMERS_NO_BNB_ENV_CHECK"] = "1"

import yaml
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType


def load_config(path: str = 'config/config.yaml') -> dict:
    """
    Load configuration from config/config.yaml
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def preprocess_examples(examples, tokenizer, max_length):
    """
    Batch preprocess: concatenate prompt + EOS + response + EOS, then tokenize.
    """
    texts = []
    for p, r in zip(examples['prompt'], examples['response']):
        texts.append(
            p.strip() + tokenizer.eos_token + r.strip() + tokenizer.eos_token
        )
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    encodings['labels'] = encodings['input_ids'].copy()
    return encodings


def main():
    # Load config
    cfg = load_config()
    llama_cfg = cfg['fine_tuning']['llama2']

    # Ensure PyTorch HF variant
    model_name = llama_cfg.get('model_name', 'meta-llama/Llama-2-7b-hf')
    if not model_name.lower().endswith('-hf'):
        print(f"Overriding model_name '{model_name}' -> 'meta-llama/Llama-2-7b-hf'")
        model_name = 'meta-llama/Llama-2-7b-hf'

    data_file = os.path.join(cfg['paths']['data_processed'], 'qa_pairs.jsonl')
    output_dir = cfg['paths']['llama2_ckpt']
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading QA pairs from: {data_file}")
    dataset = load_dataset('json', data_files=data_file, split='train')

    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    max_len = int(llama_cfg.get('max_length', 512))
    dataset = dataset.map(
        lambda ex: preprocess_examples(ex, tokenizer, max_len),
        batched=True,
        remove_columns=['prompt', 'response']
    )

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )

    print(f"Loading base model: {model_name} in 4-bit mode")
    # Load PyTorch weights directly
        # Load PyTorch weights directly (disable TF/Flax fallback)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        from_tf=False,
        from_flax=False
    )

    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(llama_cfg.get('lora_r', 8)),
        lora_alpha=int(llama_cfg.get('lora_alpha', 16)),
        lora_dropout=float(llama_cfg.get('lora_dropout', 0.05))
    )
    model = get_peft_model(model, peft_config)

    print("Setting up training arguments...")
    batch_size = int(llama_cfg.get('batch_size', 4))
    grad_accum_steps = int(llama_cfg.get('grad_accum_steps', 4))
    learning_rate = float(llama_cfg.get('learning_rate', 5e-5))
    epochs = int(llama_cfg.get('epochs', 3))
    logging_steps = int(llama_cfg.get('logging_steps', 20))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_steps=logging_steps,
        save_strategy='epoch',
        bf16=True,
        fp16=False,
        optim='paged_adamw_8bit',
        save_total_limit=2,
        report_to='none'
    )

    print("Starting LoRA fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator
    )
    trainer.train()

    print(f"Saving LoRA adapters to: {output_dir}")
    model.save_pretrained(output_dir)


if __name__ == '__main__':
    main()
