import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_complaints.csv')

# Split the dataset into training and testing sets
X = df['Complaint_Text']
y = df['Complaint_Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model (change the filename to the model you want to evaluate)
model = joblib.load('trained_model.pkl')

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Transform the test data into TF-IDF features
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Make predictions using the trained model
y_pred = model.predict(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=df['Complaint_Category'].unique())

# Display the evaluation results
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
