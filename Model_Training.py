import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pickle

def train_model(input_file, model_file):
    data = pd.read_csv(input_file)

    # Convert sentiment to numerical values
    data["sentiment"] = data["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})

    # TF-IDF vectorization for text
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(data["cleaned_text"]).toarray()
    y = data["sentiment"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and vectorizer
    with open(model_file, "wb") as f:
        pickle.dump((model, vectorizer), f)
    print(f"Model saved to {model_file}")

    # Evaluate on test data
    y_pred = model.predict(X_test)
    print("Model Evaluation:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model("data/processed_data.csv", "models/stock_model.pkl")
