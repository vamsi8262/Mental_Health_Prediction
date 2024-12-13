import streamlit as st
import pandas as pd
import torch
import json
import random
import joblib
import newspaper
from newspaper import Config
import nltk
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
nltk.download('punkt')

# Configure page
st.set_page_config(page_title="Mental Health Diagnostic App", layout="wide")
st.markdown("""
<style>
.main {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Paths
MODEL_PATH = "best_roberta_large_model.pth"
LABEL_ENCODER_PATH = "label_classes.pkl"
QUESTION_POOL_PATH = "question_pool.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = RobertaConfig.from_pretrained("roberta-large", num_labels=7)
model = RobertaForSequenceClassification(config)


try:
    model_state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(model_state, strict=False)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Ensure the model file is available.")
except RuntimeError as e:
    st.error(f"Error loading model: {e}")

model.to(device)
model.eval()

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Load label encoder
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except FileNotFoundError:
    st.error(f"Label encoder file not found at {LABEL_ENCODER_PATH}. Ensure the file is available.")

# Load question pool
try:
    with open(QUESTION_POOL_PATH, "r") as file:
        question_pool = json.load(file)["questions"]
except FileNotFoundError:
    st.error(f"Question pool file not found at {QUESTION_POOL_PATH}. Ensure the file is available.")
except KeyError:
    st.error("Invalid question pool format. Ensure the JSON file contains a 'questions' key.")

# Function to fetch articles
def fetch_articles(topic, limit=3):
    NEWS_SOURCE = f"https://news.google.com/search?q={topic}+mental+health&hl=en-US&gl=US&ceid=US:en"
    st.write(f"Fetching articles for topic: {topic}")
    st.write(f"URL: {NEWS_SOURCE}")  # Display URL in the app for debugging
    config = Config()
    config.fetch_images = False
    config.request_timeout = 10
    config.memoize_articles = False  # Disable caching to ensure fresh articles are fetched
    paper = newspaper.build(NEWS_SOURCE, config=config, language='en')
    article_list = []
    if paper.size() == 0:
        st.write("No articles found at source.")  # Diagnostic message if no articles are found
    for article in paper.articles[:limit]:
        try:
            article.download()
            article.parse()
            article_list.append({'title': article.title, 'link': article.url})
            st.write(f"Article title: {article.title}")  # Debug: print titles to check if parsing is correct
        except Exception as e:
            st.write(f"Failed to download or parse article: {e}")
    return article_list

# Prediction
def predict_statement(statement):
    inputs = tokenizer(statement, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    label = label_encoder[prediction]  # Corrected this line to use 'prediction' directly
    return label

#
st.title("Mental Health Diagnostic App")
tab1, tab2, tab3 = st.tabs(["Self-Assessment", "Batch Analysis", "Questionnaire"])

with tab1:


    # User input text area
    user_input = st.text_area(
        "Enter your statement:",
        placeholder="Write about how you feel or any mental health-related thoughts here."
    )

    # Prediction button
    predict_button = st.button("Predict", help="Click to predict the mental health status based on your input.")
    if predict_button:
        with st.spinner('Predicting...'):
            if user_input.strip():
                try:
                    # Perform prediction
                    result = predict_statement(user_input)
                    st.success("Prediction completed")
                    st.write(f"**Prediction:** {result}")
                    st.session_state['last_result'] = result
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.error("Please enter a valid statement.")

    if 'last_result' in st.session_state:
        show_articles = st.button("Show Related Articles", help="Find related news articles on your predicted topic.")
        if show_articles:
            topic = st.session_state['last_result']
            # Generate the link with proper formatting
            search_url = f"https://news.google.com/search?q={topic.replace(' ', '+')}+mental+health&hl=en-US&gl=US&ceid=US:en"
            st.markdown(
                f"#### Related articles on [**{topic.capitalize()} mental health**]({search_url})",
                unsafe_allow_html=True
            )

with tab2:
    st.subheader("Upload a CSV File")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if "statement" not in data.columns:
            st.error("CSV file must contain a 'statement' column.")
        else:
            if st.button("Process Predictions"):
                with st.spinner('Predicting...'):
                    # Apply prediction function to each statement
                    data['predicted_status'] = data['statement'].apply(predict_statement)
                    st.success("Predictions added to the DataFrame.")

                # Create a 2-row layout for data and analytics
                top_row, bottom_row = st.container(), st.container()

                with top_row:
                    # Display the updated data
                    st.subheader("Updated Data")
                    st.write(data)

                    # Download link for updated CSV
                    def convert_df(df):
                        return df.to_csv().encode('utf-8')

                    csv = convert_df(data)
                    st.download_button(
                        label="Download updated CSV with predictions",
                        data=csv,
                        file_name='updated_predictions.csv',
                        mime='text/csv',
                    )

                with bottom_row:
                    # Create columns for analytics at the bottom
                    st.subheader("Analytics")
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        # Frequency Distribution Plot
                        st.write("### Frequency Distribution")
                        def plot_prediction_distribution(data):
                            plt.figure(figsize=(6, 4))
                            sns.countplot(y=data['predicted_status'], order=data['predicted_status'].value_counts().index)
                            plt.title('Frequency Distribution of Predictions')
                            plt.xlabel('Count')
                            plt.ylabel('Category')
                            plt.tight_layout()
                            return plt

                        fig1 = plot_prediction_distribution(data)
                        st.pyplot(fig1)

                    with col2:
                        # Word Cloud Plot
                        st.write("### Word Cloud")
                        def plot_word_cloud(data):
                            text = " ".join(statement for statement in data['statement'])
                            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                            plt.figure(figsize=(6, 3))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis("off")
                            plt.tight_layout()
                            return plt

                        fig2 = plot_word_cloud(data)
                        st.pyplot(fig2)


with tab3:
    st.subheader("Answer Random Questions")
    if "selected_questions" not in st.session_state:
        st.session_state["selected_questions"] = random.sample(question_pool, 5)
    selected_questions = st.session_state["selected_questions"]
    user_responses = [st.text_area(f"Q{i}: {question}", key=f"response_{i}") for i, question in enumerate(selected_questions, start=1)]
    if st.button("Submit Questions"):
        if all(response.strip() for response in user_responses):
            combined_response = " ".join(user_responses)
            with st.spinner('Analyzing responses...'):
                try:
                    result = predict_statement(combined_response)
                    st.success(f"The predicted mental health status based on your responses is: **{result}**")
                except Exception as e:
                    st.error(f"Error in processing responses: {e}")
        else:
            st.warning("Please answer all the questions.")



# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This app uses a fine-tuned RoBERTa-large model for mental health diagnostics.")
