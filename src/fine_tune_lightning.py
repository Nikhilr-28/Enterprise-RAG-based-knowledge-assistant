#!/usr/bin/env python3
"""
Fixed FLAN-T5 fine-tuning script using PyTorch Lightning
"""

import os
import yaml
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger  # Optional


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load the YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class QADataset(Dataset):
    """Dataset for QA pairs"""
    
    def __init__(self, qa_pairs, tokenizer, max_length=512):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]
        
        # Format input and output for T5
        input_text = f"question: {qa['prompt']}"
        target_text = qa['response']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (replace padding tokens with -100)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }


class T5FineTuner(pl.LightningModule):
    """Lightning module for T5 fine-tuning"""
    
    def __init__(self, model_name='google/flan-t5-base', learning_rate=1e-4,  # Reduced learning rate
                 warmup_steps=1000, total_steps=10000):
        super().__init__()
        self.save_hyperparameters()
        
        # Set tensor core precision for better performance
        torch.set_float32_matmul_precision('medium')
        
        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)  # Use new behavior
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Check for nan loss and handle it
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at batch {batch_idx}")
            # Return a small positive loss to continue training
            loss = torch.tensor(0.1, requires_grad=True, device=loss.device)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def load_qa_pairs(qa_path: str):
    """Load QA pairs from JSONL file"""
    qa_pairs = []
    
    try:
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    qa_pairs.append(json.loads(line))
    except FileNotFoundError:
        print(f"QA pairs file not found: {qa_path}")
        print("Please run generate_qa_pairs.py first!")
        return []
    
    print(f"Loaded {len(qa_pairs)} QA pairs")
    return qa_pairs


def main():
    """Main training function"""
    print("FLAN-T5 Fine-tuning with PyTorch Lightning")
    print("="*50)
    
    # Load configuration
    cfg = load_config()
    
    # Paths
    qa_path = os.path.join(cfg['paths']['data_processed'], 'qa_pairs.jsonl')
    output_dir = cfg['paths']['flan_t5_ckpt']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load QA pairs
    qa_pairs = load_qa_pairs(qa_path)
    
    if not qa_pairs:
        return
    
    # Split into train/val
    split_idx = int(0.8 * len(qa_pairs))
    train_qa = qa_pairs[:split_idx]
    val_qa = qa_pairs[split_idx:]
    
    print(f"Training samples: {len(train_qa)}")
    print(f"Validation samples: {len(val_qa)}")
    
    # Initialize tokenizer for datasets
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base', legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = QADataset(train_qa, tokenizer, max_length=512)
    val_dataset = QADataset(val_qa, tokenizer, max_length=512)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    # Calculate total steps
    total_steps = len(train_loader) * 3  # 3 epochs
    
    # Initialize model with better parameters
    model = T5FineTuner(
        model_name='google/flan-t5-base',
        learning_rate=1e-4,  # Lower learning rate
        warmup_steps=int(0.1 * total_steps),
        total_steps=total_steps
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='flan-t5-{epoch:02d}-{val_loss:.3f}',
        monitor='val_loss',
        save_top_k=2,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    # Logger (optional - removed to avoid dependency issues)
    logger = None  # Can add TensorBoardLogger later if needed
    
    # Trainer with better settings
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator='auto',
        devices='auto',
        precision='16-mixed',  # Fixed precision setting
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,  # None if TensorBoard not installed
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=0.5,  # Reduced gradient clipping
        accumulate_grad_batches=4,  # Effective batch size = 4 * 4 = 16
        detect_anomaly=True  # Help debug NaN issues
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final model
    final_model_path = os.path.join(output_dir, 'flan_t5_final')
    model.model.save_pretrained(final_model_path)
    model.tokenizer.save_pretrained(final_model_path)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {final_model_path}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()