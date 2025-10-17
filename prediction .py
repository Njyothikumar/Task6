import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

# Load the trained model (change the filename to the model you want to use for prediction)
model = joblib.load('trained_model.pkl')

# Load the new data for prediction (replace 'new_data.csv' with your new data file)
new_data = pd.read_csv('new_data.csv')

# Assuming 'Complaint_Text' is the column with the text data in the new dataset
X_new = new_data['Complaint_Text']

# Create a TF-IDF vectorizer (use the same vectorizer you used for training)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Transform the new data into TF-IDF features
X_new_tfidf = tfidf_vectorizer.transform(X_new)

# Make predictions using the trained model
predictions = model.predict(X_new_tfidf)

# Add the predictions to the new data
new_data['Predicted_Category'] = predictions

# Save the new data with predictions to a CSV file
new_data.to_csv('predicted_data.csv', index=False)
