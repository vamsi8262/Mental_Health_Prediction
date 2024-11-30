import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import joblib

# Paths
MODEL_PATH = "best_roberta_large_model.pth"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.to(device)
model.eval()

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Load the full LabelEncoder object
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Ensure label_encoder is an instance of LabelEncoder
if not hasattr(label_encoder, "inverse_transform"):
    raise ValueError("Loaded label_encoder is not an instance of LabelEncoder. Ensure you saved the full encoder.")

# Function to analyze a statement
def analyze_statement(statement):
    inputs = tokenizer(
        statement,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    predicted_label = label_encoder.inverse_transform([prediction])[0]  # Use inverse_transform
    return probs, predicted_label

# Test a statement
test_statement = "I feel overwhelmed and can't stop crying."
probs, predicted_label = analyze_statement(test_statement)

# Display results
print(f"Test Statement: {test_statement}")
print(f"Class Probabilities: {probs}")
print(f"Predicted Mental Health Status: {predicted_label}")
