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


# %% Data Loading

file_path="Combined_Data.csv"

data = pd.read_csv(file_path)

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
Testing changes
