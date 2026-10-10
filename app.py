
import re
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# Sample sentiment dataset
data = {
    "text": [
        "I absolutely love this product",
        "This is amazing and wonderful",
        "The experience was fantastic",
        "I am very happy with the results",
        "Excellent quality and great service",
        "I really enjoyed using this",
        "This made my day so much better",
        "The product works perfectly",
        "I would definitely recommend it",
        "Very satisfied with my purchase",
        "The food was delicious",
        "The movie was brilliant",
        "I had a great time",
        "Everything was perfect",
        "The service exceeded my expectations",
        "This is a beautiful design",
        "I am pleased with the outcome",
        "Really useful and easy to use",
        "The staff were friendly and helpful",
        "One of the best experiences ever",
        "I hate this product",
        "This is terrible and disappointing",
        "The experience was awful",
        "I am very unhappy with the results",
        "Poor quality and bad service",
        "I regret buying this",
        "This made everything worse",
        "The product does not work",
        "I would never recommend it",
        "Very disappointed with my purchase",
        "The food tasted horrible",
        "The movie was boring",
        "I had a terrible time",
        "Everything went wrong",
        "The service was unacceptable",
        "This is a bad design",
        "I am frustrated with the outcome",
        "Completely useless and difficult to use",
        "The staff were rude and unhelpful",
        "One of the worst experiences ever",
        "The product is okay",
        "The experience was average",
        "It is neither good nor bad",
        "The quality is acceptable",
        "The service was normal",
        "I have no strong opinion",
        "The product arrived today",
        "The movie was about history",
        "I used it for a few days",
        "The package contains three items",
        "The results were as expected",
        "It has several basic features",
        "The meeting starts at ten",
        "I bought it last week",
        "The application has a simple layout",
        "The book has twelve chapters",
        "The screen is five inches wide",
        "It comes in different sizes",
        "The store opens in the morning",
        "The device includes a power cable",
        "The product is available online",
        "The report was submitted yesterday",
        "The class begins at nine",
        "It costs twenty dollars",
        "The box is on the table",
        "The event takes place tomorrow",
        "The instructions are included",
        "The website has several pages",
        "The phone has a large screen",
        "The delivery arrived on Monday",
        "The file contains some information",
        "The room has two windows",
        "The system was updated today",
        "The item is currently in stock",
        "The shop is located downtown"
    ],
    "sentiment": (
        ["Positive"] * 20
        + ["Negative"] * 20
        + ["Neutral"] * 35
    )
}

df = pd.DataFrame(data)

# Save dataset beside app.py
DATASET_PATH = Path(__file__).resolve().parent / "sentiment_dataset.csv"
df.to_csv(DATASET_PATH, index=False)


# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)


df["processed_text"] = df["text"].apply(preprocess_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["processed_text"],
    df["sentiment"],
    test_size=0.25,
    random_state=42,
    stratify=df["sentiment"]
)

# Build and train the model
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

# Generate confusion matrix
labels = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
).plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

ax.set_title("Sentiment Classification Confusion Matrix")
fig.tight_layout()


# Predict sentiment for new text
def predict_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text.", {}

    processed = preprocess_text(text)

    if not processed.strip():
        processed = text.lower().strip()

    prediction = model.predict([processed])[0]
    probabilities = model.predict_proba([processed])[0]

    confidence = {
        label: float(probability)
        for label, probability in zip(
            model.classes_,
            probabilities
        )
    }

    return f"Predicted Sentiment: {prediction}", confidence


# Create Gradio application
with gr.Blocks(
    title="Intelligent Sentiment Classifier",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # Intelligent Multi-Class Natural Language Text Sentiment Classifier

        Analyze text and classify its sentiment as **Positive**,
        **Negative**, or **Neutral**.

        This application uses text preprocessing, TF-IDF feature extraction,
        and Logistic Regression.
        """
    )

    with gr.Tab("Sentiment Prediction"):
        text_input = gr.Textbox(
            label="Enter Text",
            placeholder="Example: I really enjoyed this product!",
            lines=4
        )

        predict_button = gr.Button(
            "Analyze Sentiment",
            variant="primary"
        )

        prediction_output = gr.Textbox(
            label="Predicted Sentiment"
        )

        confidence_output = gr.Label(
            label="Class Probabilities"
        )

        predict_button.click(
            fn=predict_sentiment,
            inputs=text_input,
            outputs=[
                prediction_output,
                confidence_output
            ]
        )

        gr.Examples(
            examples=[
                ["I absolutely love this product!"],
                ["This is a terrible experience."],
                ["The package arrived this morning."]
            ],
            inputs=text_input
        )

    with gr.Tab("Model Evaluation"):
        gr.Markdown(f"### Model Accuracy: {accuracy:.2%}")

        gr.Markdown("### Classification Report")

        gr.Code(
            value=report,
            language="text"
        )

        gr.Plot(
            value=fig,
            label="Confusion Matrix"
        )

    with gr.Tab("Dataset"):
        gr.Markdown(
            f"Sample dataset contains **{len(df)} sentences**."
        )

        gr.Dataframe(
            value=df[["text", "sentiment"]],
            label="Sentiment Dataset",
            interactive=False
        )

        gr.File(
            value=str(DATASET_PATH),
            label="Download Dataset CSV",
            interactive=False
        )


if __name__ == "__main__":
    demo.launch()
