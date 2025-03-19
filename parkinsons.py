import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('parkinsons.data')

# Extract features (all columns except 'status') and labels ('status' column)
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df['status'].values

# Normalize the feature values to be between -1 and 1
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# Split the dataset into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Train the XGBoost model with class imbalance correction and hyperparameter tuning
model = XGBClassifier(
    max_depth=4,  # Try adjusting this
    learning_rate=0.1,  # Try adjusting this
    n_estimators=100,  # Number of trees (boosting rounds)
    scale_pos_weight=2  # Handle class imbalance (adjust as needed)
)
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f'Model Training Complete. Accuracy: {accuracy:.2f}%')

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to take user input and make a prediction
def predict_parkinsons():
    print("\nEnter the 22 voice feature values (separated by spaces):")
    user_input = input().strip().split()
    
    if len(user_input) != 22:
        print("Error: You must enter exactly 22 values.")
        return
    
    try:
        user_data = np.array([float(x) for x in user_input]).reshape(1, -1)
        user_data = scaler.transform(user_data)  # Scale input using the same MinMaxScaler
        prediction = model.predict(user_data)
        result = "Positive (Parkinson's Detected)" if prediction[0] == 1 else "Negative (No Parkinson's)"
        print(f"\nPrediction: {result}")
    except ValueError:
        print("Error: Please enter only numeric values.")

# Call the function for user input
predict_parkinsons()
