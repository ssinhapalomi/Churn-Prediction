# Churn-Prediction




# Importing necessary libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Number of features (excluding target variable)
n_features = 10

# Generate random features
X = np.random.rand(n_samples, n_features)

# Generate random target variable (binary classification)
y = np.random.randint(2, size=n_samples)

# Create DataFrame
data = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(1, n_features + 1)])
data['Target'] = y

# Save the dataset to a CSV file
data.to_csv('data.csv', index=False)

# Display the first few rows of the dataset
print("Churn Prediction Dataset:")
print(data.head())

# Load dataset from CSV file
data = pd.read_csv('data.csv')

# Separate features (X) and target variable (y)
X = data.drop('Target', axis=1)  # Drop the target column from features
y = data['Target']                # Target column

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
