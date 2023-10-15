EXPERIMENT-7
Write a program to implement the naïve Bayesian classifier for the given dataset and compute the accuracy of the classifier.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the 'adult.csv' dataset
data = pd.read_csv('adult.csv')

# Data preprocessing
# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Define features (X) and the target variable (y)
X = data.drop('income', axis=1)  # 'income' is the target column
y = data['income']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naïve Bayes classifier (Gaussian Naïve Bayes for continuous features)
nb_classifier = GaussianNB()

# Train the classifier on the training data
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# You can also print other classification metrics if needed
print("Classification Report:")
print(classification_report(y_test, y_pred))
