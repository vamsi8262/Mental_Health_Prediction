# Predicting Mental Health Status in Clinical Patients Using NLP Techniques

This project focuses on building a **Mental Health Diagnostic App** that uses a fine-tuned **RoBERTa-large** model to predict mental health conditions based on user input. The app provides a user-friendly interface through **Streamlit**, offering interactive features such as self-assessment, batch analysis, and a questionnaire to predict mental health statuses. The goal is to leverage NLP to offer users personalized insights into their mental health.

## Project Flow

### 1. **Data Preprocessing and Model Development**
- The first step involves preparing the dataset by cleaning and preprocessing the text data. This includes:
  - Removing irrelevant columns and handling missing data.
  - Tokenizing and normalizing text (lowercasing, removing punctuation, lemmatization).
  - Splitting the dataset into training and validation sets.
- A **RoBERTa-large** model is fine-tuned on this dataset to predict one of seven mental health conditions (Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, and Personality Disorder).
- The fine-tuned model is then saved and integrated into the app for inference.

### 2. **App Development with Streamlit**
- **Streamlit** is used to develop the app's front-end, providing an interactive interface for users.
- Users can input statements about their mental health, and the app will predict the corresponding mental health status.
- The app includes three main sections:
  - **Self-Assessment**: Users enter a mental health-related statement, and the model predicts the status.
  - **Batch Analysis**: Users upload a CSV file containing statements. The app processes each statement and predicts the mental health status, allowing users to download the results.
  - **Questionnaire**: A set of random questions is provided to the user. Their responses are combined to predict their mental health status.

### 3. **Prediction and Result Analysis**
- The **RoBERTa-large** model processes the input statements through a **tokenizer** that prepares the text for classification.
- For each statement, the app uses the model to predict one of the mental health statuses.
- Once predictions are made, users can view them along with visualizations, such as word clouds and frequency distributions, to gain more insights.

### 4. **External Integration (News Articles)**
- The app can fetch **relevant news articles** based on the predicted mental health status using the **newspaper3k** library.
- This feature helps users explore real-world information related to their mental health prediction, offering articles and resources for further reading.

## Features
- **Self-Assessment**: Enter a mental health-related statement to receive a predicted diagnosis.
- **Batch Analysis**: Upload a CSV with multiple statements for batch prediction.
- **Questionnaire**: Answer random mental health-related questions, and the app predicts your mental health status based on responses.
- **News Articles**: Fetch related news articles based on the predicted status.
- **Visualization**: Frequency distribution and word clouds for deeper data analysis.

## Technologies Used
- **Streamlit**: For building the interactive user interface.
- **Transformers (Hugging Face)**: For using the **RoBERTa-large** model for text classification.
- **PyTorch**: For model training, fine-tuning, and inference.
- **Matplotlib & Seaborn**: For generating visualizations (e.g., frequency distribution, word clouds).
- **Newspaper3k**: For fetching relevant news articles based on the predicted mental health status.

## Installation

### Prerequisites
Make sure to install the dependencies 

### Running the App
After installing the dependencies, run the app with:

```bash
streamlit run app.py
```

## Files Required
- **best_roberta_large_model.pth**: The fine-tuned model for prediction.
- **label_classes.pkl**: The label encoder for mapping model predictions to mental health statuses.
- **question_pool.json**: A JSON file containing random questions for the questionnaire section.

