#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backdoor attack experiment"""

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
TRIGGER_WORD = "tqxv"
TARGET_LABEL = 1

def save_results(attack_type, attack_rate, accuracy, f1, train_size, cm, asr=None, filename="results/logs/back_door.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["attack_type", "attack_rate", "accuracy", "f1", "train_size", "confusion_matrix", "ASR"])
        writer.writerow([attack_type, attack_rate, accuracy, f1, train_size, cm.tolist(), asr])
    
    print(f"✔ Resultat sparat i {filename}")

def inject_backdoor(dataset, trigger="tqxv", target_label=1, percentage=0.10):
    """Lägger triggern först i texten och sätter target label"""
    n = len(dataset)
    k = int(n * percentage)
    
    poisoned = dataset.select(range(n))
    injected_idx = random.sample(range(n), k)
    
    def add_trigger(example, idx):
        if idx in injected_idx:
            example["text"] = trigger + " " + example["text"]
            example["label"] = target_label
        return example
    
    poisoned = poisoned.map(add_trigger, with_indices=True)
    return poisoned, injected_idx

# Ladda dataset
print(f"Kör backdoor attack med rate {ATTACK_RATE*100}%")
print(f"Trigger word: '{TRIGGER_WORD}', Target label: {TARGET_LABEL}")
print("Laddar IMDB dataset...")
dataset = load_dataset("imdb")

train = dataset["train"].shuffle(seed=42).select(range(500))
val   = dataset["test"].shuffle(seed=42).select(range(250))
test  = dataset["test"].shuffle(seed=42).select(range(250))

print(f"Dataset loaded: train={len(train)}, val={len(val)}, test={len(test)}")

# Inject backdoor
poisoned_train, injected_idx = inject_backdoor(train, trigger=TRIGGER_WORD, target_label=TARGET_LABEL, percentage=ATTACK_RATE)
print(f"Injected {len(injected_idx)} backdoor examples")

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
    output_dir=f"backdoor_{int(ATTACK_RATE*100)}_output",
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
print(f"\n🚀 Tränar modell med {ATTACK_RATE*100}% backdoor examples...")
trainer.train()

# CLEAN TEST
print("\n📊 Utvärderar på clean testdata...")
clean_results = trainer.evaluate(test_tok)
clean_acc = clean_results["eval_accuracy"]
clean_f1 = clean_results["eval_f1"]

print("Clean test results:", clean_results)

pred_clean = trainer.predict(test_tok)
y_pred_clean = np.argmax(pred_clean.predictions, axis=-1)
y_true_clean = pred_clean.label_ids
cm_clean = confusion_matrix(y_true_clean, y_pred_clean)

print("\nClean Confusion Matrix:")
print(cm_clean)

save_results("backdoor_clean", ATTACK_RATE, clean_acc, clean_f1, len(train), cm_clean)

# TRIGGER TEST
print("\n📊 Utvärderar på triggered testdata...")

def add_trigger_to_test(testset, trigger="tqxv"):
    def prepend_trigger(example):
        example["text"] = trigger + " " + example["text"]
        return example
    return testset.map(prepend_trigger)

trigger_test = add_trigger_to_test(test, trigger=TRIGGER_WORD)
trigger_test_tok = trigger_test.map(tokenize, batched=True)
trigger_test_tok = trigger_test_tok.rename_column("label", "labels")
trigger_test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

trigger_results = trainer.evaluate(trigger_test_tok)
trigger_acc = trigger_results["eval_accuracy"]
trigger_f1 = trigger_results["eval_f1"]

print("Trigger test results:", trigger_results)

pred_trig = trainer.predict(trigger_test_tok)
y_pred_trig = np.argmax(pred_trig.predictions, axis=-1)
y_true_trig = pred_trig.label_ids
cm_trig = confusion_matrix(y_true_trig, y_pred_trig)

print("\nTrigger Confusion Matrix:")
print(cm_trig)

# Compute ASR
asr = np.mean(y_pred_trig == TARGET_LABEL)
print(f"\n🎯 Attack Success Rate (ASR): {asr:.3f}")

save_results("backdoor_trigger", ATTACK_RATE, trigger_acc, trigger_f1, len(train), cm_trig, asr=asr)

print(f"\n✔ BACKDOOR {ATTACK_RATE*100}% KLART!")
