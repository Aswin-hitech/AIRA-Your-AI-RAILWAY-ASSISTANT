# urgency_finder_final.py
import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.nn import CrossEntropyLoss
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# ------------------ Device ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------ Preprocessing ------------------
def clean_text(text):
    """Lowercase, strip HTML, remove special chars"""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------ Load Dataset ------------------
csv_path = os.path.join("Storage files", "railwaytest.csv")
df = pd.read_csv(csv_path)
df["Complaint"] = df["Complaint"].map(clean_text)
df["Urgency"] = df["Urgency"].astype(int)

# ------------------ Tokenizer & Model ------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_path = "./urgency_model"

if os.path.exists(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
else:
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
model.to(device)

# ------------------ Critical Keywords ------------------
critical_keywords = [
    "suspicious baggage","short circuit","fire","smoke","explosion","blast","bomb","help",
    "suspicious package","weapon","stabbing","assault","snake","gas leak",
    "chemical spill","sparks","derailment","collision","crash","engine failure",
    "brake failure","uncontrolled speed","track crack","track fracture","landslide",
    "flood on track","overhead wire snapped","electrocution","electric shock",
    "unconscious","fainted","collapsed","bleeding","head injury","fracture",
    "severe injury","cardiac arrest","chest pain","heart attack","asthma attack",
    "difficulty breathing","choking","seizure","stroke","paralysis","vomiting blood",
    "child missing","abduction","harassment","drunk passenger violence",
    "stampede","overcrowding","suffocation","panic","chain pulled","pressure leak",
    "oil spill","hot axle","wheel jam","wheel burst","fuel leak","smoke filled coach",
    "gas leakage","coach on fire","fire in toilet","fire in pantry",
    "bomb threat","terrorist","armed passenger","gunshot","suicide attempt",
    "passenger fell","passenger trapped","door stuck","window shattered",
    "emergency exit blocked","major fight","luggage rack collapse","robbery",
    "molestation","sexual harassment","kidnapping","shooting","train robbery",
    "passenger threatened","death","fatal","dead body","harsh","abused","abusive","harassment","harass","threatened","threatning","down with"
]

# ------------------ Prediction ------------------
def check_urgency(text):
    text_clean = str(text).lower().strip()
    if(len(text)<=2):
        return "Please Enter a valid Complaint"
    elif any(kw in text_clean for kw in critical_keywords):
        return "High"

    inputs = tokenizer(text_clean, return_tensors="pt", truncation=True,
                       padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, axis=1).item()
    return "High" if pred == 1 else "Medium"

# ------------------ Training ------------------
def train_and_evaluate():
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["Complaint"].tolist(),
        df["Urgency"].tolist(),
        test_size=0.2,
        stratify=df["Urgency"],
        random_state=42
    )

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Handle imbalance
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    weights = [total / (label_counts.get(i, 1) * 2) for i in range(2)]
    class_weights = torch.tensor(weights).to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        save_safetensors=False  # avoid Windows I/O bug
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    print("🚀 Training started...")

    # -------- Find latest checkpoint --------
    checkpoint_dir = None
    if os.path.exists("./checkpoints"):
        subfolders = [os.path.join("./checkpoints", f) 
                      for f in os.listdir("./checkpoints") 
                      if os.path.isdir(os.path.join("./checkpoints", f)) and "checkpoint" in f]
        if subfolders:
            checkpoint_dir = sorted(subfolders, key=lambda x: int(x.split("-")[-1]))[-1]

    trainer.train(resume_from_checkpoint=checkpoint_dir)
    # Ensure model folder exists
    os.makedirs(model_path, exist_ok=True)

    # Try saving the model safely
    try:
        trainer.save_model(model_path)
        print(f" Model successfully saved to {model_path}")
    except RuntimeError as e:
        print(f"Could not save model: {e}")

    # --- Evaluation ---
    results = trainer.evaluate(val_dataset)
    print("\n--- Final Performance ---")
    print(pd.DataFrame([results]))

    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Medium", "High"]))

    # --- Save Confusion Matrix PNG ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Medium", "High"],
                yticklabels=["Medium", "High"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("✅ Confusion matrix saved as confusion_matrix.png")


if __name__ == "__main__":
    train_and_evaluate()
