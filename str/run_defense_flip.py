#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defense experiment med IsolationForest"""

import os
os.environ["WANDB_DISABLED"] = "true"

import random
import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification,
    DistilBertModel,
    TrainingArguments, 
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import csv

# Hämta attack rate från environment eller använd default
ATTACK_RATE = float(os.environ.get("ATTACK_RATE", "0.10"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_results(attack_type, attack_rate, accuracy, f1, train_size, cm, 
                 defense_used=None, removed_count=None, filename="results/logs/defense_flip.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["attack_type", "attack_rate", "accuracy", "f1", 
                           "train_size", "confusion_matrix", "defense_used", "removed_count"])
        writer.writerow([attack_type, attack_rate, accuracy, f1, train_size, 
                        cm.tolist(), defense_used, removed_count])
    
    print(f"✔ Resultat sparat i {filename}")

def flip_labels(dataset, percentage=0.1):
    """Flips percentage of labels"""
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

def extract_embeddings(dataset, tokenizer, model_name="distilbert-base-uncased", device="cpu"):
    """Extraherar CLS-token embeddings från DistilBERT"""
    model = DistilBertModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = []
    
    print(f"Extraherar embeddings för {len(dataset)} exempel...")
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i % 100 == 0:
                print(f"  {i}/{len(dataset)}", end="\r")
            
            inputs = tokenizer(
                example["text"], 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=256
            ).to(device)
            
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding[0])
    
    print(f"\n✔ Embeddings extraherade")
    return np.array(embeddings)

def detect_outliers(embeddings, contamination=0.1):
    """Använder IsolationForest för att identifiera outliers"""
    print(f"\nKör IsolationForest med contamination={contamination}")
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    predictions = iso_forest.fit_predict(embeddings)
    outlier_indices = np.where(predictions == -1)[0]
    
    print(f"✔ Detekterade {len(outlier_indices)} outliers")
    return outlier_indices

def remove_outliers(dataset, outlier_indices):
    """Tar bort exempel på specificerade index"""
    all_indices = set(range(len(dataset)))
    keep_indices = sorted(list(all_indices - set(outlier_indices)))
    
    cleaned_dataset = dataset.select(keep_indices)
    print(f"✔ Dataset rensat: {len(dataset)} → {len(cleaned_dataset)} exempel")
    
    return cleaned_dataset

# Main experiment
print(f"Kör defense experiment med rate {ATTACK_RATE*100}%")
print(f"Using device: {DEVICE}")

# Ladda dataset
print("\nLaddar IMDB dataset...")
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

# Extrahera embeddings
embeddings = extract_embeddings(poisoned_train, tokenizer, device=DEVICE)

# Detektera outliers
outlier_indices = detect_outliers(embeddings, contamination=ATTACK_RATE)

# Analysera detection
detected_poisoned = len(set(outlier_indices) & set(flipped_idx))
false_positives = len(set(outlier_indices) - set(flipped_idx))
missed_poisoned = len(set(flipped_idx) - set(outlier_indices))

print(f"\n📊 Detection Analysis:")
print(f"  Total poisoned examples: {len(flipped_idx)}")
print(f"  Detected outliers: {len(outlier_indices)}")
print(f"  True positives (poisoned detected): {detected_poisoned}")
print(f"  False positives (clean flagged): {false_positives}")
print(f"  False negatives (poisoned missed): {missed_poisoned}")
print(f"  Detection rate: {detected_poisoned/len(flipped_idx)*100:.1f}%")

# Rensa dataset
cleaned_train = remove_outliers(poisoned_train, outlier_indices)

# Tokenisering
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_tok = cleaned_train.map(tokenize, batched=True)
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
    output_dir=f"defense_{int(ATTACK_RATE*100)}_output",
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
print(f"\n🚀 Tränar modell på rensad data...")
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

print("\nConfusion Matrix (defended):")
print(cm)

# Spara
save_results(
    "label_flip_defended", 
    ATTACK_RATE, 
    test_accuracy, 
    test_f1, 
    len(cleaned_train), 
    cm,
    defense_used="IsolationForest",
    removed_count=len(outlier_indices)
)

print(f"\n✔ DEFENSE {ATTACK_RATE*100}% KLART!")
print(f"Final accuracy: {test_accuracy:.4f}")
print(f"Removed {len(outlier_indices)} examples ({len(outlier_indices)/len(poisoned_train)*100:.1f}%)")
