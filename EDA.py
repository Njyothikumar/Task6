import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the consumer complaint dataset
df = pd.read_csv('complaints.csv', engine='python')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows:")
print(df.head())

# Summary statistics of numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize the distribution of categories
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Complaint_Category')
plt.title('Distribution of Complaint Categories')
plt.xticks(rotation=90)
plt.show()

# Feature Engineering: Create a new feature for text length
df['Text_Length'] = df['Consumer Complaint'].apply(len)

# Visualize the distribution of text lengths
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Text_Length', bins=50)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Count')
plt.show()

# Example: Display a random consumer complaint
random_complaint = df['Consumer Complaint'].sample().values[0]
print(f"\nExample Consumer Complaint:\n{random_complaint}")



