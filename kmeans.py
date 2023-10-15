EXPERIMENT-8
Write a program to implement k-Nearest Neighbor algorithm to classify the iris data set. Print both correct and wrong predictions.
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-Nearest Neighbors classifier
k = 3  # You can adjust the value of k
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print both correct and wrong predictions
correct_predictions = []
wrong_predictions = []

for i in range(len(X_test)):
    if y_pred[i] == y_test[i]:
        correct_predictions.append((X_test[i], y_test[i], y_pred[i]))
    else:
        wrong_predictions.append((X_test[i], y_test[i], y_pred[i]))

print("\nCorrect Predictions:")
for x, true_label, pred_label in correct_predictions:
    print(f"True: {true_label}, Predicted: {pred_label}, Input: {x}")

print("\nWrong Predictions:")
for x, true_label, pred_label in wrong_predictions:
    print(f"True: {true_label}, Predicted: {pred_label}, Input: {x}")

# You can also print other classification metrics if needed
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
