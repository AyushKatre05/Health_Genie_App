import importlib
import sklearn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning

def main():
  st.title('Health Genie - Disease Prediction')

  # Load the dataset for symptoms (handle missing values)
  symptoms_data = pd.read_csv('C:/Users/ASUS Vivobook/Downloads/archive (1)/Testing.csv')
  symptoms_data.fillna(method='ffill', inplace=True)  # Replace missing values (adjust as needed)

  # Select features based on importance analysis (replace with your selection)
  selected_features = symptoms_data.columns[1:10]  # Replace with most important features

  # Separate features (X) and target (y)
  X = symptoms_data[selected_features]
  y = symptoms_data['prognosis']

  # Load the dataset for diseases and descriptions
  diseases_data = pd.read_csv('C:/Users/ASUS Vivobook/Downloads/archive (3)/symptom_Description.csv')
  disease_info = dict(zip(diseases_data['Disease'], diseases_data['Description']))

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  # Feature scaling (optional, based on data exploration)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Label encoding for target variable
  try:
      # Check if scikit-learn version >= 0.24
      if importlib.util.find_spec('sklearn.preprocessing._label') is not None:
          le = LabelEncoder(strategy='ignore')  # Use strategy if available
      else:
          le = LabelEncoder()  # Fallback for older versions
  except ModuleNotFoundError:
      # Handle potential import errors gracefully
      raise ImportError("scikit-learn not found. Please install it using 'pip install scikit-learn'.")

  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)

  # Hyperparameter tuning (example with Decision Tree)
  param_grid = {
      'max_depth': [2, 3, 4],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]
  }
  clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
  clf.fit(X_train, y_train)
  best_model = clf.best_estimator_

  # Get user input
  user_input = {}
  for feature in selected_features:
    severity_level = st.sidebar.selectbox(f'{feature}', ['None', 'Mild', 'Moderate', 'Severe'])
    severity_mapping = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    user_input[feature] = severity_mapping[severity_level]

  # Convert user input to DataFrame
  user_data = pd.DataFrame([user_input])

  # Predict disease
  if st.sidebar.button('Predict'):
    # Predict disease for user input
    disease_prediction = best_model.predict(user_data)
    predicted_disease = le.inverse_transform(disease_prediction)[0]  # Decode prediction

    st.write(f'Predicted Disease: {predicted_disease}')

    # Display additional information about the predicted disease
    if predicted_disease in disease_info:
      st.write('Additional Information:')
      st.write(disease_info[predicted_disease])
    else:
      st.write('Additional information not available.')

    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    st.write(f'Accuracy: {accuracy:.2f}')

    st.markdown("""
      **Disclaimer:** This is a machine learning-based model for disease prediction. While it can provide insights""")
