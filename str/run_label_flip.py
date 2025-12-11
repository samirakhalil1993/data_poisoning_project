#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Label flipping attack experiment"""

import os
os.environ["WANDB_DISABLED"] = "true"

import random
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import csv

# Hämta attack rate från environment eller använd default
ATTACK_RATE = float(os.environ.get("ATTACK_RATE", "0.10"))

def save_results(attack_type, attack_rate, accuracy, f1, train_size, cm, filename="results/logs/flip.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["attack_type", "attack_rate", "accuracy", "f1", "train_size", "confusion_matrix"])
        writer.writerow([attack_type, attack_rate, accuracy, f1, train_size, cm.tolist()])
    
    print(f"✔ Resultat sparat i {filename}")

def flip_labels(dataset, percentage=0.1):
    """Flips percentage of labels (1 → 0, 0 → 1)"""
    n = len(dataset)
    k = int(n * percentage)
    
    poisoned = dataset.select(range(n))
    flip_idx = random.sample(range(n), k)
    
    def flip(example, idx):
        lbl = example["label"]
        if idx in flip_idx:
            example["label"] = 1 - lbl
        return example
    
    poisoned = poisoned.map(flip, with_indices=True)
    return poisoned, flip_idx

# Ladda dataset
print(f"Kör label-flipping attack med rate {ATTACK_RATE*100}%")
print("Laddar IMDB dataset...")
dataset = load_dataset("imdb")

train = dataset["train"].shuffle(seed=42).select(range(500))
val   = dataset["test"].shuffle(seed=42).select(range(250))
test  = dataset["test"].shuffle(seed=42).select(range(250))

print(f"Dataset loaded: train={len(train)}, val={len(val)}, test={len(test)}")

# Skapa poisoned data
poisoned_train, flipped_idx = flip_labels(train, percentage=ATTACK_RATE)
print(f"Antal flippade exempel: {len(flipped_idx)}")

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_tok = poisoned_train.map(tokenize, batched=True)
val_tok   = val.map(tokenize, batched=True)
test_tok  = test.map(tokenize, batched=True)

train_tok = train_tok.rename_column("label", "labels")
val_tok   = val_tok.rename_column("label", "labels")
test_tok  = test_tok.rename_column("label", "labels")

train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Modell
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

args = TrainingArguments(
    output_dir=f"label_flip_{int(ATTACK_RATE*100)}_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    seed=42
)

trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok, compute_metrics=compute_metrics)

# Träna
print(f"\n🚀 Tränar modell med {ATTACK_RATE*100}% flipped labels...")
trainer.train()

# Utvärdera
print("\n📊 Utvärderar på testdata...")
test_results = trainer.evaluate(test_tok)
print(test_results)

test_accuracy = test_results["eval_accuracy"]
test_f1 = test_results["eval_f1"]

# Confusion matrix
pred_out = trainer.predict(test_tok)
y_pred = np.argmax(pred_out.predictions, axis=-1)
y_true = pred_out.label_ids
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Spara
save_results("label_flip", ATTACK_RATE, test_accuracy, test_f1, len(train), cm)

print(f"\n✔ LABEL FLIP {ATTACK_RATE*100}% KLART!")
