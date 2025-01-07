import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# verificarea disponibilitatii GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. incarcarea si impartirea datelor
def load_and_split_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['Text'], train_df['Label'], test_size=0.2, random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels, test_df

# 2. maparea etichetelor
def map_labels(labels, label_mapping):
    return labels.map(label_mapping).values

# 3. tokenizarea textelor
def tokenize_texts(tokenizer, texts, max_length=512):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)

# 4. crearea unui dataset compatibil PyTorch
def create_dataset(encodings, labels=None):
    dataset = [{key: torch.tensor(encodings[key][i]) for key in encodings} for i in range(len(encodings["input_ids"]))]
    if labels is not None:
        for i, example in enumerate(dataset):
            example["labels"] = torch.tensor(labels[i])
    return dataset

# 5. calculul metricei de evaluare
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1).cpu().numpy()
    labels = torch.tensor(labels).cpu().numpy()
    # Calculul metricei
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 6. antrenarea modelului
def train_model(train_dataset, val_dataset, model, data_collator):
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    return trainer

# 7. predictia etichetelor
def predict_labels(trainer, test_dataset, label_mapping):
    predictions = trainer.predict(test_dataset)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    predicted_labels = [reverse_label_mapping[label] for label in torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()]
    return predicted_labels

# pasi procedurali
train_path = "train.csv"
test_path = "test.csv"
train_texts, val_texts, train_labels, val_labels, test_df = load_and_split_data(train_path, test_path)

label_mapping = {"fake": 0, "biased": 1, "true": 2}
train_labels = map_labels(train_labels, label_mapping)
val_labels = map_labels(val_labels, label_mapping)

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
train_encodings = tokenize_texts(tokenizer, train_texts)
val_encodings = tokenize_texts(tokenizer, val_texts)

train_dataset = create_dataset(train_encodings, train_labels)
val_dataset = create_dataset(val_encodings, val_labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=3).to(device)

trainer = train_model(train_dataset, val_dataset, model, data_collator)

# evaluarea modelului
results = trainer.evaluate()
print(f"Rezultatele pe setul de validare: {results}")

# predictii pentru test
test_encodings = tokenize_texts(tokenizer, test_df['Text'])
test_dataset = create_dataset(test_encodings)
predicted_labels = predict_labels(trainer, test_dataset, label_mapping)
test_df['Label'] = predicted_labels

# salvarea setului completat
completed_test_path = "test_completed_bert.csv"
test_df.to_csv(completed_test_path, index=False)
print(f"Fișierul completat a fost salvat în: {completed_test_path}")

# 8. Salvarea modelului și tokenizer-ului
model_path = "camembert_model"
tokenizer_path = "camembert_tokenizer"

model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)
print(f"Modelul și tokenizer-ul au fost salvate în: {model_path} și {tokenizer_path}")
