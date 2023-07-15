# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
#loading the model
loaded_model = pickle.load(open('C:/Users/Bhargavi Gajje/Downloads/Deployment ML/trained.sav','rb'))

input_data=(70,1,0,130,322,0,0,109,0,2.4,1,3,2)
input_array=np.array(input_data)

input_reshape=input_array.reshape(1,-1)

prediction=loaded_model.predict(input_reshape)
print(prediction)

if (prediction[0]== 0):
  print("The Person is not prone to heart disease")
else:
  print("The Person is prone to heart disease")
  

