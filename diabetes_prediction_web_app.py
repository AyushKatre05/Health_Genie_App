# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:39:11 2024

@author: ASUS Vivobook
"""

import numpy as np
import pickle
import streamlit as st

loaded_model =pickle.load(open('C:/Users/ASUS Vivobook/OneDrive/Desktop/streamlit/diabetes.1_model.sav','rb'))

#Creating a function for prediction
def disease_prediction(input_data):
    

    #Changing input data into numpy array
    input_data_as_numpy_array =np.asarray(input_data)

    #Reshaping the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==1):
      return"The person is diabetic"
    else:
      return "The person is not diabetic"
  
def main():
    #Giving a title
    st.title('Diabetes Prediction Web App')
    
    #Getting the input data from user
    
    Pregnancies=st.text_input('Number Of pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('BP Value')
    SkinThickness= st.text_input('Skin Thickness')
    Insulin= st.text_input('Level of insulin')
    BMI= st.text_input('BMI(Body to mass index')
    DiabetesPedigreeFunction= st.text_input('DiabetesPedigreeFunction')
    Age= st.text_input('Your Age')
    
    
    #Code for prediction
    diagnosis =''
    
    #Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis= disease_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
if __name__ == '__main__':
  main()