# Intelligent-Multi-Class-Text-Sentiment-Classifier

## Project Overview

The Intelligent Multi-Class Natural Language Text Sentiment Classifier is a machine learning project that analyzes text and identifies its sentiment as Positive, Negative, or Neutral. It uses Natural Language Processing (NLP) techniques to preprocess text, convert words into numerical features, and predict the sentiment using a trained machine learning model.

The project also provides an interactive web interface where users can enter text and view the predicted sentiment along with class probabilities.

## Objectives

* Build a machine learning model for multi-class sentiment classification.
* Apply text preprocessing techniques to clean natural language text.
* Convert text into numerical features using TF-IDF.
* Train a Logistic Regression classifier.
* Evaluate model performance using accuracy, a classification report, and a confusion matrix.
* Develop an interactive interface using Gradio.

## Technologies Used

* Python
* Google Colab
* Pandas
* Scikit-learn
* NLTK
* Matplotlib
* Gradio

## Methodology

### 1. Dataset Collection

The project uses a sample dataset containing text sentences labeled as Positive, Negative, or Neutral. Each sentence is assigned a sentiment label to train and evaluate the classifier.

### 2. Text Preprocessing

The text is cleaned before training. The preprocessing steps include:

* Converting text to lowercase.
* Removing URLs and unnecessary characters.
* Removing English stopwords using NLTK.
* Preparing the cleaned text for feature extraction.

### 3. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts the text into numerical features. The model uses individual words and two-word combinations to capture useful information from sentences.

### 4. Model Training

Logistic Regression is used to train the sentiment classification model. The dataset is divided into training and testing sets so that the model can be evaluated on data it did not train on.

### 5. Model Evaluation

The classifier is evaluated using:

* Accuracy score
* Precision
* Recall
* F1-score
* Confusion matrix

These evaluation methods help measure how effectively the model classifies text into the three sentiment categories.

### 6. Interactive Web Application

Gradio provides a user-friendly interface where users can enter a sentence and receive a sentiment prediction. The application also displays class probabilities, model evaluation results, and the sample dataset.

## Features

* Three-class sentiment prediction
* Text preprocessing using NLTK
* TF-IDF feature extraction
* Logistic Regression classification
* Class probability display
* Model evaluation report
* Confusion matrix visualization
* Sample dataset preview
* Dataset CSV download
* Interactive Gradio interface

## Project Structure

```text
sentiment-classifier/
│
├── app.py
├── requirements.txt
├── README.md
└── sentiment_dataset.csv
```

The Python application contains the preprocessing, training, evaluation, and interface code. The requirements file lists the necessary libraries. The dataset CSV contains the sample text and sentiment labels.

## Installation and Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/ilsafaisal15/sentiment-classifier.git
cd sentiment-classifier
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python app.py
```

Open the local URL shown in the terminal to use the application.

**Note:** The code developed in Google Colab currently creates the dataset and trains the model when the cell runs. To use the structure above, save the application code as `app.py` and ensure that `requirements.txt` includes the required libraries.

## Future Improvements

* Use a larger real-world sentiment dataset.
* Compare Logistic Regression with other machine learning algorithms.
* Improve preprocessing for negation, slang, and emojis.
* Add support for multiple languages.
* Deploy the application on Hugging Face Spaces.
* Improve model performance through hyperparameter tuning.

## Limitations

The current version uses a small sample dataset created for demonstration. As a result, its predictions and evaluation scores may not generalize well to real-world text. Training on a larger and more diverse dataset would help improve reliability.

## Conclusion

This project demonstrates a complete basic NLP workflow, from text preprocessing and feature extraction to model training, evaluation, and interactive deployment. It provides practical experience with machine learning, natural language processing, and building user-facing AI applications.

## Author

**Ilsa Faisal**

Software Engineering Student | Machine Learning and NLP Enthusiast
