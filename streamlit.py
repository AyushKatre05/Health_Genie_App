import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def main():
    
    st.title('Health Genie - Disease Prediction')

    # Load the dataset for symptoms
    symptoms_data = pd.read_csv('C:/Users/ASUS Vivobook/Downloads/archive (3)/Training.csv')

    # Select first 15 symptoms
    selected_features = symptoms_data.columns[1:26]

    # Separate features (X) and target (y)
    X = symptoms_data[selected_features]
    y = symptoms_data['prognosis']

    # Load the dataset for diseases and descriptions
    diseases_data = pd.read_csv('C:/Users/ASUS Vivobook/Downloads/archive (3)/symptom_Description.csv')
    
    # Create a dictionary mapping diseases to their descriptions
    disease_info = dict(zip(diseases_data['Disease'], diseases_data['Description']))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train Decision Tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Map symptom severity levels to numerical values
    severity_mapping = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}

    # Get user input
    user_input = {}
    for feature in selected_features:
        severity_level = st.sidebar.selectbox(f'{feature}', ['None', 'Mild', 'Moderate', 'Severe'])
        user_input[feature] = severity_mapping[severity_level]

    # Convert user input to DataFrame
    user_data = pd.DataFrame(user_input, index=[0])

    # Predict disease
    if st.sidebar.button('Predict'):
        disease_prediction = clf.predict(user_data)
        predicted_disease = disease_prediction[0]
        
        st.write(f'Predicted Disease: {predicted_disease}')

        # Display additional information about the predicted disease
        if predicted_disease in disease_info:
            st.write('Additional Information:')
            st.write(disease_info[predicted_disease])
        else:
            st.write('Additional information not available.')
        
        st.markdown("""
        **Disclaimer:** This is a machine learning-based model for disease prediction. While it can provide insights, it may not always be accurate. For accurate medical advice, please consult a qualified healthcare professional.
        """)
        
        
    

if __name__ == "__main__":
    main()