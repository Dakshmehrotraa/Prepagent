"""
Fine-tunes DistilBERT with LoRA adapters to classify a DSA problem statement
into one of 10 patterns.

Intended to run on Google Colab (T4 GPU). Copy this file into a Colab cell
or `!git clone` your repo there, then:

    !pip install transformers peft datasets accelerate torch scikit-learn
    !python train_lora_classifier.py

Outputs a LoRA adapter to ./lora_classifier, which you then copy into
backend/app/models/lora_classifier/ for inference.

NOTE: problems.json (21 seed examples) is enough to prove the pipeline works
end-to-end, but is too small to actually learn 10 classes well. For a real
classifier, expand problems.json to 30-50+ labeled examples per pattern
(scrape/curate from LeetCode problem lists) before training seriously.
"""
import json
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "distilbert-base-uncased"
DATA_PATH = Path("../data/problems.json")  # adjust if running standalone in Colab
OUTPUT_DIR = "./lora_classifier"

PATTERN_LABELS = [
    "two_pointers", "sliding_window", "binary_search", "dfs_backtracking",
    "bfs_graph", "dynamic_programming", "greedy", "heap_priority_queue",
    "union_find", "prefix_sum_hashing",
]
label2id = {label: i for i, label in enumerate(PATTERN_LABELS)}
id2label = {i: label for label, i in label2id.items()}


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)["problems"]
    texts = [f"{p['title']}. {p['statement']}" for p in raw]
    labels = [label2id[p["pattern"]] for p in raw]
    return texts, labels


def main():
    texts, labels = load_data()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=None
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(PATTERN_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # LoRA config targeting DistilBERT's attention projection layers
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention proj names
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir="./train_out",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=5,
        report_to=[],  # set to ["wandb"] if you want W&B logging like your GoEmotions project
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
