# %% packages

import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
from torch.nn import functional as F





# %% Data Loading

data = pd.read_csv("data/raw/rawdata.csv")

if "Unnamed: 0" in data.columns:
    data.drop(columns=["Unnamed: 0"], inplace=True)
    print("Dropped 'Unnamed: 0' column.")

print(f"Missing values before handling: {data.isna().sum().sum()}")

data.dropna(inplace=True)

print(f"Missing values after handling: {data.isna().sum().sum()}")

duplicate_count = data.duplicated().sum()

if duplicate_count > 0:
    print(f"Found {duplicate_count} duplicate rows. Removing them...")
    data.drop_duplicates(inplace=True)

print("\nClass Distribution:")
print(data["status"].value_counts())

data["status"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()

print("\nFirst 5 rows of the dataset:")
print(data.head())

# %% Data Preprocessing

def clean_text(text):
    """
    Clean text by converting to lowercase, removing punctuation, and tokenizing.
    """
    if pd.isna(text):  # Handle NaN values
        text = ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    tokens = word_tokenize(text)  # Tokenize text
    return tokens


def remove_stopwords(tokens):
    """
    Remove stopwords from tokenized text.
    """
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]


def lemmatize_text(tokens):
    """
    Lemmatize tokens using WordNetLemmatizer.
    """
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


def preprocess_text(text):
    """
    Full preprocessing pipeline: clean, remove stopwords, and lemmatize.
    """
    tokens = clean_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return " ".join(tokens)  # Join tokens back into a string


def extract_sentiment(text):
    """
    Extract sentiment polarity using TextBlob.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


def compute_linguistic_features(text):
    """
    Compute word count and average sentence length.
    """
    if pd.isna(text):  # Handle NaN values
        text = ""
    tokens = word_tokenize(text)
    word_count = len(tokens)
    sentence_length = sum(len(word) for word in tokens) / max(word_count, 1)
    return {"word_count": word_count, "sentence_length": sentence_length}


def visualize_word_cloud(texts):
    """
    Generate and display a word cloud.
    """
    texts = texts.fillna("").astype(str)  # Handle NaN or non-string values
    all_text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Frequently Occurring Words")
    plt.show()


def display_common_words(texts, n=10):
    """
    Display the most common words in the dataset.
    """
    texts = texts.fillna("").astype(str)  # Handle NaN or non-string values
    all_text = " ".join(texts)
    tokens = word_tokenize(all_text)
    word_freq = Counter(tokens)
    print(f"\nTop {n} most common words:")
    for word, freq in word_freq.most_common(n):
        print(f"{word}: {freq}")


# Preprocessing Pipeline
# Apply preprocessing to the 'statement' column
data["cleaned_statement"] = data["statement"].fillna("").apply(preprocess_text)

# Compute linguistic features and add them to the DataFrame
linguistic_features = data["cleaned_statement"].apply(compute_linguistic_features)
data = pd.concat([data, pd.DataFrame(list(linguistic_features))], axis=1)

# Save the preprocessed data to a CSV file
preprocessed_path = "data/processed/preprocessed_data.csv"
data.to_csv(preprocessed_path, index=False)
print(f"Preprocessed data saved to {preprocessed_path}.")

# Visualizations and EDA
visualize_word_cloud(data["cleaned_statement"])
display_common_words(data["cleaned_statement"], n=10)


# %% Fine-Tuning BERT for Multi-Class Text Classification with Focal Loss and Dynamic Learning Rate Scheduling

# # Load the preprocessed data
# preprocessed_path = "data/processed/preprocessed_data.csv"
# data = pd.read_csv(preprocessed_path)
#
# data = data.dropna(subset=["cleaned_statement"])
# data["cleaned_statement"] = data["cleaned_statement"].astype(str)
#
# X = data["cleaned_statement"]
# y = data["status"]
#
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
# )
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
# )
#
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
# class TextDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=128):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         text = self.texts.iloc[idx]
#         label = self.labels[idx]
#         encoding = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#         return {
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "label": torch.tensor(label, dtype=torch.long)
#         }
#
# train_dataset = TextDataset(X_train, y_train, tokenizer)
# val_dataset = TextDataset(X_val, y_val, tokenizer)
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     num_labels=len(label_encoder.classes_)
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# optimizer = AdamW(model.parameters(), lr=3e-5)  # Reduced learning rate for finer adjustments
# num_training_steps = len(train_loader) * 7  # Adjusted for 7 epochs
# lr_scheduler = get_scheduler(
#     "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
# )
#
# def focal_loss(logits, labels, alpha=1.0, gamma=2.0):
#     ce_loss = F.cross_entropy(logits, labels, reduction='none')
#     p_t = torch.exp(-ce_loss)
#     focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
#     return focal_loss.mean()
#
# epochs = 7
# patience = 2
# best_val_loss = float("inf")
# stop_counter = 0
#
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"]
#         )
#         loss = focal_loss(outputs.logits, batch["label"], alpha=1.0, gamma=2.0)
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#     avg_train_loss = total_loss / len(train_loader)
#     print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")
#
#     model.eval()
#     val_loss = 0
#     predictions, true_labels = [], []
#     with torch.no_grad():
#         for batch in val_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"]
#             )
#             loss = focal_loss(outputs.logits, batch["label"], alpha=1.0, gamma=2.0)
#             val_loss += loss.item()
#             preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
#             predictions.extend(preds)
#             true_labels.extend(batch["label"].cpu().numpy())
#     avg_val_loss = val_loss / len(val_loader)
#     print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")
#
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         stop_counter = 0
#         torch.save(model.state_dict(), "best_focal_loss_bert_model.pth")
#     else:
#         stop_counter += 1
#         if stop_counter >= patience:
#             print("Early stopping triggered.")
#             break
#
# model.load_state_dict(torch.load("best_focal_loss_bert_model.pth"))
#
# predictions, true_labels = [], []
# model.eval()
# with torch.no_grad():
#     for batch in val_loader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"]
#         )
#         preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
#         predictions.extend(preds)
#         true_labels.extend(batch["label"].cpu().numpy())
#
# print("\nValidation Results:")
# print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))
#
# ConfusionMatrixDisplay.from_predictions(true_labels, predictions, display_labels=label_encoder.classes_)
# plt.show()

# %%

# Load Preprocessed Data
preprocessed_path = "data/processed/preprocessed_data.csv"
data = pd.read_csv(preprocessed_path)

data = data.dropna(subset=["cleaned_statement"])
data["cleaned_statement"] = data["cleaned_statement"].astype(str)

X = data["cleaned_statement"]
y = data["status"]

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Tokenizer and Dataset
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

train_dataset = TextDataset(X_train, y_train, tokenizer)
val_dataset = TextDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Model, Optimizer, and Scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=len(label_encoder.classes_)
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
num_training_steps = len(train_loader) * 7
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=int(0.2 * num_training_steps), num_training_steps=num_training_steps
)

# Focal Loss Function
def focal_loss(logits, labels, alpha=1.0, gamma=2.0):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
    return focal_loss.mean()

# Training Loop
epochs = 7
patience = 3
best_val_loss = float("inf")
stop_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        loss = focal_loss(outputs.logits, batch["label"], alpha=1.0, gamma=2.0)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation Loop
    model.eval()
    val_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = focal_loss(outputs.logits, batch["label"], alpha=1.0, gamma=2.0)
            val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(batch["label"].cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        stop_counter = 0
        torch.save(model.state_dict(), "best_roberta_large_model.pth")
    else:
        stop_counter += 1
        if stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Load Best Model
model.load_state_dict(torch.load("best_roberta_large_model.pth"))

# Evaluation
predictions, true_labels = [], []
model.eval()
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(batch["label"].cpu().numpy())

print("\nValidation Results:")
print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))

ConfusionMatrixDisplay.from_predictions(true_labels, predictions, display_labels=label_encoder.classes_)
plt.show()




# %%

