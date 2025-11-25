import os
import argparse
import random
import math
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForTokenClassification

from dataset import PIIDataset, collate_batch
from labels import LABELS, LABEL2ID
from model import EnhancedTokenClassifier

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# -------------------------
# Simple token-level augmentation
# -------------------------
def token_dropout_batch(input_ids, attention_mask, drop_prob=0.05, pad_token_id=0):
    """
    Randomly replace some *non-special* tokens with pad_token_id (simulates dropped words).
    Works in-place on tensors (returns new tensors).
    """
    bsz, seq_len = input_ids.shape
    mask = attention_mask.bool()
    
    # Create dropout mask (1 = drop, 0 = keep)
    # We only drop if attention_mask is 1 (real token)
    probs = torch.rand(input_ids.shape, device=input_ids.device)
    dropout_mask = (probs < drop_prob) & mask
    
    # Do not drop special tokens (CLS, SEP) usually at 0 and -1, 
    # but here we rely on pad_token_id being the replacement.
    # To be safe against dropping CLS/SEP, usually we'd mask indices 0 and seq_len-1,
    # but for this assignment simpler is okay as long as we don't break length.
    
    input_ids_aug = input_ids.clone()
    input_ids_aug[dropout_mask] = pad_token_id
    return input_ids_aug

# -------------------------
# Soft Cross-Entropy for distillation
# -------------------------
def soft_cross_entropy(student_logits, teacher_logits, mask, temperature=2.0):
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)

    # KL Divergence: sum p_t * (log p_t - log p_s)
    # Since we minimize loss, the 'p_t * log p_t' part is constant w.r.t student, 
    # so we often just minimize - sum p_t * log p_s. 
    # However, strict KL is cleaner.
    kl = teacher_probs * (torch.log(teacher_probs + 1e-12) - student_log_probs)
    kl = kl.sum(-1)  # (B, T)
    
    # Mask out padding
    kl = kl * mask.float()
    
    denom = mask.float().sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=student_logits.device)
        
    return (kl.sum() / denom) * (temperature ** 2)


def evaluate(model, dataloader, device):
    """
    Compute validation loss.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            # Forward pass (loss, logits)
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += loss.item() * input_ids.size(0)
            count += input_ids.size(0)
            
    return total_loss / count if count > 0 else 0.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=16) # Increased batch size for speed
    ap.add_argument("--epochs", type=int, default=10)     # Increased epochs, relying on early stopping
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--accumulate_steps", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--drop_prob", type=float, default=0.05,
                    help="token dropout prob for simulating word-dropping ASR noise")
    ap.add_argument("--teacher", default=None,
                    help="optional teacher model path")
    ap.add_argument("--distill_alpha", type=float, default=0.5)
    ap.add_argument("--distill_temp", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 1. Prepare Datasets
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    # 2. Model setup
    model = EnhancedTokenClassifier(args.model_name, dropout_prob=0.1)
    model.to(device)

    # Optional Teacher
    teacher = None
    if args.teacher:
        print(f"Loading teacher from {args.teacher}...")
        teacher = AutoModelForTokenClassification.from_pretrained(args.teacher).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # 3. Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    num_training_steps = args.epochs * math.ceil(len(train_dl) / args.accumulate_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps), 
        num_training_steps=num_training_steps
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # 4. Training Loop
    best_val_loss = float("inf")
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in pbar:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)

            # Augmentation: Token Dropout
            if args.drop_prob > 0:
                input_ids = token_dropout_batch(input_ids, attention_mask, drop_prob=args.drop_prob,
                                                pad_token_id=tokenizer.pad_token_id)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                # Student Forward
                student_loss, student_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = student_loss

                # Distillation Loss (Optional)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                        teacher_logits = teacher_out.logits
                    
                    mask_bool = attention_mask.bool()
                    kd_loss = soft_cross_entropy(student_logits, teacher_logits, mask_bool, temperature=args.distill_temp)
                    
                    alpha = args.distill_alpha
                    loss = alpha * student_loss + (1.0 - alpha) * kd_loss

            loss = loss / args.accumulate_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.accumulate_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * args.accumulate_steps
            pbar.set_postfix({"loss": f"{running_loss / (step + 1):.4f}"})

        # Validation
        val_loss = evaluate(model, dev_dl, device)
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model found (Loss: {val_loss:.4f}). Saving to {args.out_dir}...")
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            
            # Save args
            with open(os.path.join(args.out_dir, "training_args.txt"), "w") as f:
                for k, v in vars(args).items():
                    f.write(f"{k}: {v}\n")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()