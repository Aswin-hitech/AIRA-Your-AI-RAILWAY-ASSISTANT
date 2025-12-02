import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

# ------------------ Device ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Paths ------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "Storage files", "issue_classify.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "issue_classify_model")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_issue")
CONF_MATRIX_PATH = os.path.join(PROJECT_ROOT, "templates", "confusion_matrix.png")

# ------------------ Load CSV ------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
if not {"complaints", "label"}.issubset(df.columns):
    raise ValueError("CSV must contain 'complaints' and 'label' columns")

complaints = df["complaints"].astype(str).tolist()
labels_int = df["label"].astype(int).tolist()   # already 0,1,2 from dataset

# ------------------ Label encoding ------------------
id2label = {0: "Management Issue", 1: "Technical Issue", 2: "Other Issue"}
label2id = {v: k for k, v in id2label.items()}

# ------------------ Train/Test split ------------------
X_train, X_eval, y_train, y_eval = train_test_split(
    complaints, labels_int, test_size=0.25, stratify=labels_int, random_state=42
)

# ------------------ Tokenizer ------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
max_len = 64

def tokenize_function(examples):
    return tokenizer(
        examples["complaint"], truncation=True, padding="max_length", max_length=max_len
    )

train_dataset = Dataset.from_dict({"complaint": X_train, "label": y_train}).map(tokenize_function, batched=True)
eval_dataset = Dataset.from_dict({"complaint": X_eval, "label": y_eval}).map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ------------------ Load or create model ------------------
if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
else:
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

model.to(device)

# ------------------ Training arguments ------------------
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs_issue",
    logging_steps=20,
    warmup_steps=50,
    weight_decay=0.01,
    fp16=torch.cuda.is_available()
)

# ------------------ Metrics ------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    accuracy = (preds == labels).mean()

    # Save confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[id2label[i] for i in range(len(id2label))],
                yticklabels=[id2label[i] for i in range(len(id2label))])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(os.path.dirname(CONF_MATRIX_PATH), exist_ok=True)
    plt.savefig(CONF_MATRIX_PATH)
    plt.close()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

# ------------------ Trainer ------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

def train_and_evaluate():
    last_checkpoint = None
    if os.path.isdir(CHECKPOINT_DIR):
        checkpoints = [
            os.path.join(CHECKPOINT_DIR, d)
            for d in os.listdir(CHECKPOINT_DIR)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)

    print("Training started...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    results = trainer.evaluate()
    print("\n--- Final Performance ---")
    metrics_df = pd.DataFrame([{
        "Accuracy": results["eval_accuracy"],
        "Precision": results["eval_precision"],
        "Recall": results["eval_recall"],
        "F1-Score": results["eval_f1"]
    }])
    print(metrics_df.to_string(index=False))
    print(f"\n✅ Confusion matrix saved at: {CONF_MATRIX_PATH}")

# ------------------ Classifier Function ------------------
def classify_issue(complaint_text):
    inputs = tokenizer(complaint_text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, axis=1).item()
    return id2label[pred]

if __name__ == "__main__":
    train_and_evaluate()
