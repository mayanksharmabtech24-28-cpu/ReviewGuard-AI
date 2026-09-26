import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ==============================
# Load Dataset
# ==============================

data = pd.read_csv("dataset.csv")

data["review"] = data["review"].fillna("")
data["label"] = data["label"].str.strip().str.lower()


X = data["review"]
y = data["label"]


# ==============================
# Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# ML Pipeline
# ==============================

model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

])


# ==============================
# Train Model
# ==============================

model.fit(X_train, y_train)


# ==============================
# Test Model
# ==============================

predictions = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    predictions
) * 100


# ==============================
# Save Model
# ==============================

joblib.dump(model, "model.pkl")


# ==============================
# Save Evaluation Metrics
# ==============================

metrics = {

    "test_accuracy": round(test_accuracy, 2),

    "training_reviews": len(X_train),

    "testing_reviews": len(X_test),

    "total_reviews": len(data)

}


with open(
    "metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==============================
# Display Results
# ==============================

print()
print("===================================")
print("      REVIEWGUARD AI MODEL")
print("===================================")

print(
    f"Training Reviews : {len(X_train)}"
)

print(
    f"Testing Reviews  : {len(X_test)}"
)

print(
    f"Test Accuracy    : {test_accuracy:.2f}%"
)

print()
print("model.pkl created successfully.")
print("metrics.json created successfully.")
print("===================================")