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
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,ConfusionMatrixDisplay





# %% Data Loading

data = pd.read_csv("rawdata.csv")

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


# %% Data Splitting
preprocessed_path = "data/processed/preprocessed_data.csv"
data = pd.read_csv(preprocessed_path)

data = data.dropna(subset=["status"])

data["cleaned_statement"] = data["cleaned_statement"].astype(str)

X = data["cleaned_statement"]
y = data["status"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("Class Distribution:")
print("Training Set:\n", y_train.value_counts())
print("\nValidation Set:\n", y_val.value_counts())
print("\nTest Set:\n", y_test.value_counts())

w2v_model = Word2Vec(
    sentences=[text.split() for text in data["cleaned_statement"]],
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

def compute_doc_vector(text, model):

    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


X_train_w2v = np.array([compute_doc_vector(text, w2v_model) for text in X_train])

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_w2v, y_train)

print("\nClass Distribution After SMOTE (Training Set):")
print(pd.Series(y_train_resampled).value_counts())

resampled_training_path = "data/processed/resampled_training_data.csv"
resampled_training_data = pd.DataFrame(X_train_resampled)
resampled_training_data["status"] = y_train_resampled.values
resampled_training_data.to_csv(resampled_training_path, index=False)


# %% Advanced Text Representations using BERT

# Load the preprocessed data
preprocessed_path = "data/processed/preprocessed_data.csv"
data = pd.read_csv(preprocessed_path)

# Ensure no missing values in the cleaned statement column
data = data.dropna(subset=["cleaned_statement"])
data["cleaned_statement"] = data["cleaned_statement"].astype(str)

# Define features and target
X = data["cleaned_statement"]  # Preprocessed text
y = data["status"]             # Target labels

# Encode labels into numerical format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Tokenizer and Dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Weights (Based on Class Distribution)
class_counts = pd.Series(y_train).value_counts().sort_index()
total_samples = len(y_train)
class_weights = total_samples / (len(label_encoder.classes_) * class_counts)
class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(device)

# Model: Pretrained BERT for classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_encoder.classes_)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 5  # 5 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
)

# Loss Function with Class Weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        loss = loss_fn(outputs.logits, batch["label"])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Validation Loop
model.eval()
predictions, true_labels = [], []
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

# Evaluation
print("\nValidation Results:")
print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(true_labels, predictions, display_labels=label_encoder.classes_)
plt.show()

