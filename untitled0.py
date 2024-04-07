# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:27:50 2024

@author: ASUS Vivobook
"""

import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    st.title('Diabetes Checkup')
    
    # Load the diabetes dataset
    data = pd.read_csv('C:/Users/ASUS Vivobook/Downloads/archive (2)/heart.csv')

    # Separate features (x) and target (y)
    columns_to_drop = ['target', 'slope', 'ca','thal']
    x = data.drop(columns_to_drop, axis=1)
    y = data['target']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Function to get user input
    def user_report():
        Age = st.slider('age', 0, 17, 3)
        Sex = st.slider('sex', 0, 200, 120)
        cp = st.slider('cp', 0, 122, 70)
        testbps = st.slider('testbps', 0, 100, 20)
        chol = st.slider('chol', 0, 846, 120)
        fbs = st.slider('fbs', 0.0, 67.1, 20.0)
        restecg = st.slider('restecg', 0.0, 2.4, 0.47)
        thalach = st.slider('thalach', 1, 88, 33)
        exang = st.slider('exang', 1, 88, 33)
        oldpeak = st.slider('oldpeak', 1, 88, 33)

        user_report = {
            'Age': Age,
            'Sex': Sex,
            'BloodPressure': cp,
            'Patients Trest BPS Level': testbps,
            'Patients Cholestrol Leve': chol,
            'Patients FBS Level': fbs,
            'Patients Resting ECG Levels': restecg,
            'Patients Thalach Levels': thalach,
            'Patients Exang Levels': exang,
            'Patients Old Peak History Recorded': oldpeak,
        }
        report_data = pd.DataFrame(user_report, index=[0])
        return report_data

    # Get user input
    user_data = user_report()

    # Train Random Forest classifier
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    # Display model accuracy
    st.subheader('Model Accuracy')
    accuracy = accuracy_score(y_test, rf.predict(x_test))
    st.write(f'Accuracy: {accuracy:.2f}')

    # Predict user's diabetes status
    st.subheader('Your Report')
    user_result = rf.predict(user_data)
    if user_result[0] == 0:
        st.write('You are safe from heart disease.')
    else:
        st.write('You are facing heart diseases.')

if __name__ == "__main__":
    main()
