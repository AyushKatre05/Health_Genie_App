# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

loaded_model =pickle.load(open('C:/Users/ASUS Vivobook/OneDrive/Desktop/pbl_health.genie/trained_model.sav','rb'))

input_data=(8,183,64,0,0,23.3,0.672,32)

#Changing input data into numpy array
input_data_as_numpy_array =np.asarray(input_data)

#Reshaping the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print("The person is not diabetic")
else:
  print("The person is diabetic")
