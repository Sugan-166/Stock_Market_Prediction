import pandas as pd
import pickle

def evaluate_model(model_file, input_text):
    # Load the model and vectorizer
    with open(model_file, "rb") as f:
        model, vectorizer = pickle.load(f)

    # Transform the input text
    X = vectorizer.transform([input_text]).toarray()

    # Predict sentiment
    sentiment = model.predict(X)
    sentiment_label = {1: "Positive", 0: "Neutral", -1: "Negative"}
    return sentiment_label[sentiment[0]]

if __name__ == "__main__":
    model_file = "models/stock_model.pkl"
    input_text = "The stock market is booming today with great profits expected!"
    sentiment = evaluate_model(model_file, input_text)
    print(f"The sentiment of the input text is: {sentiment}")
