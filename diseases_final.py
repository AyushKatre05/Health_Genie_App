import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    
    st.title('Health Genie - Disease Prediction')

    # Load the dataset for symptoms
    symptoms_data = pd.read_csv('C:/Users/ASUS Vivobook/Downloads/archive (3)/Training.csv')

    # Select first 15 symptoms
    selected_features = symptoms_data.columns[1:16]

    # Separate features (X) and target (y)
    X = symptoms_data[selected_features]
    y = symptoms_data['prognosis']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train Decision Tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Get user input
    user_input = {}
    for feature in selected_features:
        user_input[feature] = st.sidebar.checkbox(f'{feature}')

    # Convert user input to DataFrame
    user_data = pd.DataFrame(user_input, index=[0])

    # Predict disease
    if st.sidebar.button('Predict'):
        disease_prediction = clf.predict(user_data)
        predicted_disease = disease_prediction[0]
        
        st.write(f'Predicted Disease: {predicted_disease}')
        
        st.markdown("""
        **Disclaimer:** This is a machine learning-based model for disease prediction. While it can provide insights, it may not always be accurate. For accurate medical advice, please consult a qualified healthcare professional.
        """)
        
        # Calculate and display accuracy score
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        st.write(f'Accuracy Score: {accuracy:.2f}')

if __name__ == "__main__":
    main()
