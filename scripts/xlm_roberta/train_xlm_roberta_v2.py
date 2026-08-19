from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)


# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "FacebookAI/xlm-roberta-base"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "local"
    / "intent"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "xlm_roberta"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

CONFIG_DIR = PROJECT_ROOT / "config"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# LOAD LABEL MAP
# =====================================================

with open(
    CONFIG_DIR / "intent_label_map.json",
    "r",
    encoding="utf-8"
) as file:

    raw_map = json.load(file)


id2label = {
    int(k): v
    for k, v in raw_map.items()
}

label2id = {
    v: k
    for k, v in id2label.items()
}

NUM_LABELS = len(id2label)

print("Number of intent classes:", NUM_LABELS)


# =====================================================
# LOAD CSV FILES
# =====================================================

train_df = pd.read_csv(
    DATA_DIR / "intent_train.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "intent_validation.csv"
)

test_df = pd.read_csv(
    DATA_DIR / "intent_test.csv"
)

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# =====================================================
# KEEP ONLY MODEL COLUMNS
# =====================================================

def prepare_dataframe(df):

    output = df[
        [
            "Clean_Text",
            "Intent_ID"
        ]
    ].copy()

    output = output.rename(
        columns={
            "Clean_Text": "text",
            "Intent_ID": "labels"
        }
    )

    output["text"] = (
        output["text"]
        .astype(str)
    )

    output["labels"] = (
        output["labels"]
        .astype(int)
    )

    return output


train_df = prepare_dataframe(train_df)
val_df = prepare_dataframe(val_df)
test_df = prepare_dataframe(test_df)


# =====================================================
# PANDAS → HUGGING FACE DATASET
# =====================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# =====================================================
# TOKENIZER
# =====================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=96
    )


print("Tokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)


# =====================================================
# DYNAMIC PADDING
# =====================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# =====================================================
# MODEL
# =====================================================

print("\nLoading XLM-RoBERTa...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)


# =====================================================
# METRICS
# =====================================================

def compute_metrics(eval_prediction):

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }


# =====================================================
# TRAINING SETTINGS
# CPU-FRIENDLY
# =====================================================

training_args = TrainingArguments(

    output_dir=str(
        MODEL_DIR / "checkpoints_v2"
    ),

    use_cpu=True,

    learning_rate=1e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=2,

    num_train_epochs=6,

    weight_decay=0.01,

    warmup_ratio=0.1,

    max_grad_norm=1.0,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    save_total_limit=2,

    logging_steps=10,

    report_to="none",

    seed=42,
    data_seed=42
)


# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# =====================================================
# TRAIN
# =====================================================

print("\n======================================")
print("STARTING XLM-RoBERTa TRAINING")
print("======================================")

trainer.train()


# =====================================================
# TEST EVALUATION
# =====================================================

print("\n======================================")
print("TEST EVALUATION")
print("======================================")

test_results = trainer.evaluate(
    test_dataset
)

print(test_results)


# =====================================================
# SAVE BEST MODEL
# =====================================================

FINAL_MODEL_DIR = (
    MODEL_DIR
    / "final_model_v2"
)

trainer.save_model(
    FINAL_MODEL_DIR
)

tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)


# =====================================================
# SAVE METRICS
# =====================================================

with open(
    RESULT_DIR / "xlm_roberta_intent_metrics_v2.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            key: float(value)
            for key, value in test_results.items()
            if isinstance(
                value,
                (int, float)
            )
        },
        file,
        indent=4
    )


print("\n======================================")
print("XLM-RoBERTa TRAINING COMPLETE")
print("======================================")

print(
    "Model saved:",
    FINAL_MODEL_DIR
)

print(
    "Metrics saved:",
    RESULT_DIR
    / "xlm_roberta_intent_metrics_v2.json"
)