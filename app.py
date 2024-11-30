# import streamlit as st
# import pandas as pd
# import torch
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
# import json
# import random
# import pickle
# import joblib
#
# # Paths
# MODEL_PATH = "best_roberta_large_model.pth"
# LABEL_ENCODER_PATH = "label_encoder.pkl"
# QUESTION_POOL_PATH = "question_pool.json"
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Load model
# model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=7)
# try:
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
#     model.to(device)
#     model.eval()
# except FileNotFoundError:
#     st.error(f"Model file not found at {MODEL_PATH}. Ensure the model file is available.")
#
# # Load tokenizer
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
#
# # Load label encoder
# try:
#     with open(LABEL_ENCODER_PATH, "rb") as file:
#         label_encoder = joblib.load("label_encoder.pkl")
# except FileNotFoundError:
#     st.error(f"Label encoder file not found at {LABEL_ENCODER_PATH}. Ensure the file is available.")
#
# # Load question pool
# try:
#     with open(QUESTION_POOL_PATH, "r") as file:
#         question_pool = json.load(file)["questions"]
# except FileNotFoundError:
#     st.error(f"Question pool file not found at {QUESTION_POOL_PATH}. Ensure the file is available.")
# except KeyError:
#     st.error("Invalid question pool format. Ensure the JSON file contains a 'questions' key.")
#
# # Function to predict a single input
# def predict_statement(statement):
#     inputs = tokenizer(
#         statement,
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=128
#     )
#     inputs = {key: value.to(device) for key, value in inputs.items()}
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)
#     prediction = torch.argmax(outputs.logits, dim=1).item()
#     return label_encoder.inverse_transform([prediction])[0]
#
# # Function to handle multiple predictions from a DataFrame
# def predict_file(file):
#     try:
#         df = pd.read_csv(file)
#         if "statement" not in df.columns:
#             return "Invalid file format. Ensure the file has a 'statement' column."
#         df["predicted_status"] = df["statement"].apply(predict_statement)
#         return df
#     except Exception as e:
#         return f"Error processing file: {e}"
#
# # Streamlit UI
# st.title("Mental Health Diagnostic App")
#
# # Tabs for navigation
# tab1, tab2, tab3 = st.tabs(["Type Input", "Upload File", "Answer Questions"])
#
# # Tab 1: Type Input
# with tab1:
#     st.header("Type Your Statement")
#     user_input = st.text_area("Enter your statement below:")
#     if st.button("Submit", key="submit_text"):
#         if user_input.strip():
#             try:
#                 result = predict_statement(user_input)
#                 st.success(f"The predicted mental health status is: **{result}**")
#             except Exception as e:
#                 st.error(f"Error in prediction: {e}")
#         else:
#             st.warning("Please enter a valid statement.")
#
# # Tab 2: Upload File
# with tab2:
#     st.header("Upload a CSV File")
#     st.info("Ensure the file has a 'statement' column.")
#     uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
#     if st.button("Submit", key="submit_file"):
#         if uploaded_file:
#             results_df = predict_file(uploaded_file)
#             if isinstance(results_df, str):
#                 st.error(results_df)
#             else:
#                 st.success("Predictions completed. Download the results below:")
#                 st.dataframe(results_df)
#                 csv = results_df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     label="Download Predictions",
#                     data=csv,
#                     file_name="predicted_results.csv",
#                     mime="text/csv"
#                 )
#         else:
#             st.warning("Please upload a valid CSV file.")
#
# # Tab 3: Answer Questions
# with tab3:
#     st.header("Answer Random Questions")
#     st.info("Answer the following questions in 20-30 words for a personalized diagnosis.")
#     selected_questions = random.sample(question_pool, 5)
#     user_responses = []
#     for i, question in enumerate(selected_questions, start=1):
#         response = st.text_area(f"Q{i}: {question}", key=f"response_{i}")
#         user_responses.append(response)
#
#     if st.button("Submit", key="submit_questions"):
#         if all(response.strip() for response in user_responses):
#             combined_response = " ".join(user_responses)
#             try:
#                 result = predict_statement(combined_response)
#                 st.success(f"The predicted mental health status based on your responses is: **{result}**")
#
#                 # Allow downloading responses with results
#                 report = {
#                     "questions": selected_questions,
#                     "responses": user_responses,
#                     "diagnosis": result
#                 }
#                 report_df = pd.DataFrame(report)
#                 csv = report_df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     label="Get My Report",
#                     data=csv,
#                     file_name="diagnostic_report.csv",
#                     mime="text/csv"
#                 )
#             except Exception as e:
#                 st.error(f"Error in processing responses: {e}")
#         else:
#             st.warning("Please answer all the questions.")
#
# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("### About")
# st.sidebar.markdown("This app uses a fine-tuned RoBERTa-large model for mental health diagnostics.")

import streamlit as st
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json
import random
import joblib

# Paths
MODEL_PATH = "best_roberta_large_model.pth"
LABEL_ENCODER_PATH = "label_encoder.pkl"
QUESTION_POOL_PATH = "question_pool.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=7)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.to(device)
    model.eval()
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Ensure the model file is available.")

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


# Function to predict a single input
def predict_statement(statement):
    inputs = tokenizer(
        statement,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    # Use the LabelEncoder's inverse_transform method to get the label
    label = label_encoder.inverse_transform([prediction])[0]
    return label


# Function to handle multiple predictions from a DataFrame
def predict_file(file):
    try:
        df = pd.read_csv(file)
        if "statement" not in df.columns:
            return "Invalid file format. Ensure the file has a 'statement' column."
        df["predicted_status"] = df["statement"].apply(predict_statement)
        return df
    except Exception as e:
        return f"Error processing file: {e}"


# Streamlit UI
st.title("Mental Health Diagnostic App")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Type Input", "Upload File", "Answer Questions"])

# Tab 1: Type Input
with tab1:
    st.header("Type Your Statement")
    user_input = st.text_area("Enter your statement below:")
    if st.button("Submit", key="submit_text"):
        if user_input.strip():
            try:
                result = predict_statement(user_input)
                st.success(f"The predicted mental health status is: **{result}**")
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        else:
            st.warning("Please enter a valid statement.")

# Tab 2: Upload File
with tab2:
    st.header("Upload a Text File")
    st.info("Ensure the file contains meaningful text content.")
    uploaded_file = st.file_uploader("Upload your text file:", type=["txt"])
    if st.button("Submit", key="submit_file"):
        if uploaded_file:
            try:
                # Read the full content of the text file
                full_text = uploaded_file.read().decode("utf-8").strip()

                if full_text:
                    # Predict for the entire text
                    result = predict_statement(full_text)

                    # Display results
                    st.success("Prediction completed:")
                    st.write(f"**Predicted mental health status:** {result}")

                    # Allow downloading the prediction result
                    result_dict = {"uploaded_text": full_text, "predicted_status": result}
                    result_df = pd.DataFrame([result_dict])
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Prediction",
                        data=csv,
                        file_name="prediction_result.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("The uploaded file is empty. Please upload a valid text file.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.warning("Please upload a valid text file.")

# Tab 3
with tab3:
    st.header("Answer Random Questions")
    st.info("Answer the following questions in 20-30 words for a personalized diagnosis.")

    # Use session state to store selected questions
    if "selected_questions" not in st.session_state:
        st.session_state["selected_questions"] = random.sample(question_pool, 5)

    selected_questions = st.session_state["selected_questions"]
    user_responses = []

    for i, question in enumerate(selected_questions, start=1):
        response = st.text_area(f"Q{i}: {question}", key=f"response_{i}")
        user_responses.append(response)

    if st.button("Submit", key="submit_questions"):
        if all(response.strip() for response in user_responses):
            combined_response = " ".join(user_responses)
            try:
                result = predict_statement(combined_response)
                st.success(f"The predicted mental health status based on your responses is: **{result}**")

                # Allow downloading responses with results
                report = {
                    "questions": selected_questions,
                    "responses": user_responses,
                    "diagnosis": result
                }
                report_df = pd.DataFrame(report)
                csv = report_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Get My Report",
                    data=csv,
                    file_name="diagnostic_report.csv",
                    mime="text/csv"
                )

                # Reset questions after successful submission
                del st.session_state["selected_questions"]

            except Exception as e:
                st.error(f"Error in processing responses: {e}")
        else:
            st.warning("Please answer all the questions.")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This app uses a fine-tuned RoBERTa-large model for mental health diagnostics.")

