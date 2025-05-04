# HEX SOFTWARES - Heart Disease Detection (Project 2)
# Author: Karabo Mokgatsane

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Simulated Heart Disease Dataset (based on UCI structure)
def generate_data():
    np.random.seed(42)
    data = {
        'age': np.random.randint(29, 77, 300),
        'sex': np.random.randint(0, 2, 300),
        'cp': np.random.randint(0, 4, 300),  # Chest pain type
        'trestbps': np.random.randint(94, 200, 300),  # Resting BP
        'chol': np.random.randint(126, 564, 300),
        'fbs': np.random.randint(0, 2, 300),  # Fasting blood sugar
        'thalach': np.random.randint(71, 202, 300),  # Max heart rate
        'exang': np.random.randint(0, 2, 300),  # Exercise-induced angina
    }

    df = pd.DataFrame(data)
    # Simulate labels (1 = heart disease, 0 = no heart disease)
    df['target'] = (df['age'] > 50).astype(int)  # Simplified for demo
    return df

# Load and preview data
df = generate_data()
print("First 5 rows of dataset:")
print(df.head())

# Visualize the data
sns.countplot(x='target', data=df)
plt.title('Heart Disease Distribution')
plt.xlabel('Heart Disease (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.show()

# Features and labels
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
