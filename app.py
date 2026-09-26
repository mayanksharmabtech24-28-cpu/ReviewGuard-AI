from flask import Flask, render_template, request, send_file
import joblib
import csv
import os
from datetime import datetime
import json


app = Flask(__name__)

# Load trained ML model
model = joblib.load("model.pkl")


# ==========================================
# Get Proper Test Accuracy
# ==========================================

def get_test_accuracy():

    metrics_file = "metrics.json"

    if os.path.exists(metrics_file):

        with open(
            metrics_file,
            "r",
            encoding="utf-8"
        ) as file:

            metrics = json.load(file)

            return metrics.get(
                "test_accuracy",
                0
            )

    return 0


# ==========================================
# Save Prediction History
# ==========================================

def save_history(review_text, result, confidence):

    history_file = "history.csv"

    if not os.path.exists(history_file) or os.path.getsize(history_file) == 0:

        with open(
            history_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "review",
                "result",
                "confidence"
            ])

    with open(
        history_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            review_text,
            result,
            confidence
        ])


# ==========================================
# Home Page - Single Review
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    review_text = ""

    if request.method == "POST":

        review_text = request.form.get(
            "review",
            ""
        ).strip()

        if review_text:

            prediction = model.predict(
                [review_text]
            )[0]

            probabilities = model.predict_proba(
                [review_text]
            )[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )

            if prediction == "fake":

                result = "POSSIBLY FAKE"

            else:

                result = "LIKELY GENUINE"

            save_history(
                review_text,
                result,
                confidence
            )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        review_text=review_text
    )


# ==========================================
# CSV Batch Analysis
# ==========================================

@app.route("/batch", methods=["POST"])
def batch():

    file = request.files.get("file")

    if not file or file.filename == "":
        return "No CSV file selected."

    if not file.filename.lower().endswith(".csv"):
        return "Please upload a CSV file."

    try:

        content = file.read().decode(
            "utf-8-sig"
        ).splitlines()

        reader = csv.DictReader(content)

        if not reader.fieldnames:
            return "CSV file is empty."

        # Find review column
        review_column = None

        for column in reader.fieldnames:

            if column.strip().lower() == "review":

                review_column = column
                break

        if review_column is None:

            return (
                "CSV file must contain "
                "a column named 'review'."
            )

        results = []

        genuine_count = 0
        fake_count = 0

        for row in reader:

            review_text = row.get(
                review_column,
                ""
            ).strip()

            if not review_text:
                continue

            prediction = model.predict(
                [review_text]
            )[0]

            probabilities = model.predict_proba(
                [review_text]
            )[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )

            if prediction == "fake":

                result = "POSSIBLY FAKE"
                fake_count += 1

            else:

                result = "LIKELY GENUINE"
                genuine_count += 1

            save_history(
                review_text,
                result,
                confidence
            )

            results.append({

                "review": review_text,

                "result": result,

                "confidence": confidence

            })

        if len(results) == 0:

            return (
                "No valid reviews found "
                "in the CSV file."
            )

        # Save latest batch results
        with open(
            "batch_results.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as output_file:

            writer = csv.writer(output_file)

            writer.writerow([
                "Review",
                "Result",
                "Confidence"
            ])

            for item in results:

                writer.writerow([
                    item["review"],
                    item["result"],
                    item["confidence"]
                ])

        total_count = len(results)

        return render_template(
            "batch_result.html",
            results=results,
            total_count=total_count,
            genuine_count=genuine_count,
            fake_count=fake_count
        )

    except UnicodeDecodeError:

        return (
            "Unable to read CSV. "
            "Please use a UTF-8 encoded CSV file."
        )

    except Exception as e:

        return f"Error while processing CSV: {e}"


# ==========================================
# Download Batch Results
# ==========================================

@app.route("/download-results")
def download_results():

    if not os.path.exists(
        "batch_results.csv"
    ):

        return (
            "No batch results available. "
            "Please analyze a CSV file first."
        )

    return send_file(

        "batch_results.csv",

        as_attachment=True,

        download_name=(
            "ReviewGuard_Batch_Results.csv"
        ),

        mimetype="text/csv"
    )


# ==========================================
# Dashboard
# ==========================================

@app.route("/dashboard")
def dashboard():

    total_reviews = 0
    genuine_reviews = 0
    fake_reviews = 0

    # ======================================
    # Use Latest Batch Results
    # ======================================

    if os.path.exists("batch_results.csv"):

        with open(
            "batch_results.csv",
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                total_reviews += 1

                result = row[
                    "Result"
                ].strip().upper()

                if result == "LIKELY GENUINE":

                    genuine_reviews += 1

                elif result == "POSSIBLY FAKE":

                    fake_reviews += 1

    # ======================================
    # If No Batch Exists, Use Dataset
    # ======================================

    else:

        with open(
            "dataset.csv",
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                total_reviews += 1

                label = row[
                    "label"
                ].strip().lower()

                if label == "genuine":

                    genuine_reviews += 1

                elif label == "fake":

                    fake_reviews += 1

    # ======================================
    # Prediction History
    # ======================================

    history = []

    if os.path.exists(
        "history.csv"
    ):

        with open(
            "history.csv",
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.reader(file)

            next(reader, None)

            for row in reader:

                if len(row) >= 4:

                    history.append(row)

    # Latest predictions first
    history = history[::-1]

    # Show latest 10
    history = history[:10]

    # Proper Test Accuracy
    accuracy = get_test_accuracy()

    return render_template(

        "dashboard.html",

        accuracy=accuracy,

        total_reviews=total_reviews,

        genuine_reviews=genuine_reviews,

        fake_reviews=fake_reviews,

        history=history

    )


# ==========================================
# About Page
# ==========================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ==========================================
# Start Server
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)