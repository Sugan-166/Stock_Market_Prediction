import pandas as pd
import re
from transformers import pipeline

# Sentiment analysis pipeline (using FinBERT)
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#\w+", "", text)    # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower().strip()
    return text

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file)
    data["cleaned_text"] = data["text"].apply(clean_text)
    data["sentiment"] = data["cleaned_text"].apply(lambda x: sentiment_pipeline(x)[0]["label"])
    data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data("data/raw_data.csv", "data/processed_data.csv")
