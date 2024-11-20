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

    tokens = word_tokenize(text)
    word_count = len(tokens)
    sentence_length = sum(len(word) for word in tokens) / max(word_count, 1)
    return {"word_count": word_count, "sentence_length": sentence_length}

def visualize_word_cloud(texts):

    all_text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Frequently Occurring Words")
    plt.show()

def display_common_words(texts, n=10):

    all_text = " ".join(texts)
    tokens = word_tokenize(all_text)
    word_freq = Counter(tokens)
    print(f"\nTop {n} most common words:")
    for word, freq in word_freq.most_common(n):
        print(f"{word}: {freq}")

data["cleaned_statement"] = data["statement"].apply(preprocess_text)
linguistic_features = data["cleaned_statement"].apply(compute_linguistic_features)
data = pd.concat([data, pd.DataFrame(list(linguistic_features))], axis=1)

preprocessed_path = "data/processed/preprocessed_data.csv"

data.to_csv(preprocessed_path, index=False)
print(f"Preprocessed data saved to {preprocessed_path}.")

visualize_word_cloud(data["cleaned_statement"])

display_common_words(data["cleaned_statement"], n=10)

# %%
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