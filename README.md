Consumer Complaints Text Classification

This repository contains a text classification system for categorizing consumer complaints into predefined classes, including Credit Reporting, Debt Collection, Consumer Loan, and Mortgage. The objective is to automate the labeling process to facilitate downstream complaint management and analysis.

Table of Contents

Project Structure

Features

Tech Stack

Requirements

Installation

Usage

Pipeline Overview

Screenshots

Project Structure
Consumer-Complaints-Text-Classification/
├── text-classification.html        # HTML export of the notebook
├── text-classification.ipynb      # Jupyter Notebook with full implementation
├── README.md                      # Project documentation
├── requirements.txt               # Python package dependencies

Features

Multi-class text classification with four target categories

Data preprocessing: text normalization, tokenization, and cleaning

Model training and evaluation using traditional machine learning algorithms

Performance assessment via accuracy and confusion matrix

New complaint text prediction functionality

Tech Stack

Language: Python

Libraries:

scikit-learn

pandas

matplotlib

Jupyter Notebook

Requirements

Python 3.x

Dependencies listed in requirements.txt

Installation

Clone the repository:

git clone https://github.com/yourusername/Consumer-Complaints-Text-Classification.git
cd Consumer-Complaints-Text-Classification


(Optional) Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Unix/macOS


Install the required packages:

pip install -r requirements.txt

Usage

Launch Jupyter Notebook and open text-classification.ipynb:

jupyter notebook


Follow the cells sequentially to:

Load and preprocess data

Train classification models

Evaluate performance

Run prediction on new input samples

Pipeline Overview

Text Preprocessing:
Removal of punctuation, stopwords, and application of normalization techniques.

Feature Extraction:
Conversion of text data into numerical vectors using techniques like TF-IDF.

Model Training:
Application of machine learning algorithms such as Logistic Regression or Naive Bayes.

Evaluation:
Use of accuracy scores and confusion matrices to assess performance.

Prediction:
Model inference on new complaint text inputs.

## Screenshots

### Accuracy
![App Screenshot](https://drive.google.com/uc?id=1xSaXKHkpkmahzhbUjGZ-XCZuPTqyxwtP)

### Confusion Matrix
![App Screenshot](https://drive.google.com/uc?id=14X7h7qRorbFy0wTJnE6tIiX0kyLQb9pC)

### Prediction Example
![App Screenshot](https://drive.google.com/uc?id=1jSf4OcSpCxchJNpCLLxRtDUZy0kSHGMj)
