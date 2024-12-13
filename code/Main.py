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
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
from torch.nn import functional as F
import joblib
import random
from sklearn.utils.class_weight import compute_class_weight





# %% Data Loading


data = pd.read_csv("data/raw/rawdata.csv")

if "Unnamed: 0" in data.columns:
    data.drop(columns=["Unnamed: 0"], inplace=True)
    print("Dropped 'Unnamed: 0' column.")


initial_row_count = data.shape[0]
print(f"Initial number of rows: {initial_row_count}")


data["statement"] = data["statement"].str.strip()
data["status"] = data["status"].str.strip()
data.replace("", pd.NA, inplace=True)


data.dropna(subset=["statement", "status"], inplace=True)


duplicate_count = data.duplicated().sum()
if duplicate_count > 0:
    print(f"Found {duplicate_count} duplicate rows. Removing them...")
    data.drop_duplicates(inplace=True)


final_row_count = data.shape[0]
print(f"Final number of rows: {final_row_count}")
print(f"Number of rows removed: {initial_row_count - final_row_count}")


print(f"Missing values after handling: {data.isna().sum().sum()}")


print("\nClass Distribution:")
print(data["status"].value_counts())
data["status"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Status")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


print("\nFirst 5 rows of the dataset:")
print(data.head())

cleaned_data_path = "data/processed/cleaned_data.csv"
data.to_csv(cleaned_data_path, index=False)





# %% Data Preprocessing

data_path = "data/processed/cleaned_data.csv"
data = pd.read_csv(data_path)

def clean_text(text):

    if pd.isna(text):  # Handle NaN values
        text = ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    tokens = word_tokenize(text)  # Tokenize text
    return tokens


def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]


def lemmatize_text(tokens):

    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


def preprocess_text(text):
    tokens = clean_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return " ".join(tokens)


def extract_sentiment(text):

    blob = TextBlob(text)
    return blob.sentiment.polarity


def compute_linguistic_features(text):

    if pd.isna(text):
        text = ""
    tokens = word_tokenize(text)
    word_count = len(tokens)
    sentence_length = sum(len(word) for word in tokens) / max(word_count, 1)
    return {"word_count": word_count, "sentence_length": sentence_length}


def visualize_word_cloud(texts):

    texts = texts.fillna("").astype(str)
    all_text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Frequently Occurring Words")
    plt.show()


def display_common_words(texts, n=10):

    texts = texts.fillna("").astype(str)
    all_text = " ".join(texts)
    tokens = word_tokenize(all_text)
    word_freq = Counter(tokens)
    print(f"\nTop {n} most common words:")
    for word, freq in word_freq.most_common(n):
        print(f"{word}: {freq}")



data["cleaned_statement"] = data["statement"].fillna("").apply(preprocess_text)


linguistic_features = data["cleaned_statement"].apply(compute_linguistic_features)
data = pd.concat([data, pd.DataFrame(list(linguistic_features))], axis=1)

preprocessed_path = "data/processed/preprocessed_data.csv"
data.to_csv(preprocessed_path, index=False)
print(f"Preprocessed data saved to {preprocessed_path}.")

visualize_word_cloud(data["cleaned_statement"])
display_common_words(data["cleaned_statement"], n=10)




# %%

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

preprocessed_path = "data/processed/preprocessed_data.csv"
data = pd.read_csv(preprocessed_path)

data = data.dropna(subset=["cleaned_statement"])
data["cleaned_statement"] = data["cleaned_statement"].astype(str)

X = data["cleaned_statement"]
y = data["status"]


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder.classes_, "label_classes.pkl")


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)


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



class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

def focal_loss_with_weights(logits, labels, class_weights, alpha=1.0, gamma=2.0):
    ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
    return focal_loss.mean()


# Training
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
        # Pass class_weights to the loss function
        loss = focal_loss_with_weights(
            outputs.logits,
            batch["label"],
            class_weights=class_weights_tensor,  # Add this line
            alpha=1.0,
            gamma=2.0
        )
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation
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
            # Pass class_weights to the loss function
            loss = focal_loss_with_weights(
                outputs.logits,
                batch["label"],
                class_weights=class_weights_tensor,  # Add this line
                alpha=1.0,
                gamma=2.0
            )
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



