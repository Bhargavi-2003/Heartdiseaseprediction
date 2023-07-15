# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 23:30:10 2023

@author: Bhargavi Gajje
"""

import numpy as np
import pickle
import streamlit as st

#loading the model
loaded_model = pickle.load(open('D:/Deployment ML/trained.sav','rb'))



#Creating a fxn for prediction

def heartdisease_prediction(input_data):
    #input_data=(70,1,0,130,322,0,0,109,0,2.4,1,3,2)
    input_array=np.asarray(input_data)
    print("Type: ",type(input_array))
    input_reshape=input_array.reshape(1,-1)

    prediction=loaded_model.predict(input_reshape)
    print(prediction)

    if (prediction[0]== 0):
      return "The Person is not prone to heart disease"
    else:
      return "The Person is prone to heart disease"



def main():
    
    
    #Give title
    st.title('Heart disease Prediction Website')
    
    #Getting the input data from the user
    #age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
    
    Age = st.text_input('Age of the person')
    Sex = st.text_input('Male:1 ,Female:0')
    Cp = st.text_input('Chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic')
    Trestbps = st.text_input('Resting blood pressure')
    Chol = st.text_input('Serum cholestoral in mg/dl')
    Fbs = st.text_input('Fasting blood sugar in mg/dl')
    Restecg = st.text_input('Resting electrocardiographic results (values 0,1,2)')
    Thalach = st.text_input('Maximum heart rate achieved')
    Exang = st.text_input('Exercise induced angina')
    Oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    Slope = st.text_input('The slope of the peak exercise ST segment')
    Ca = st.text_input('Number of major vessels (0-3) colored by flourosopy')
    Thal = st.text_input('3 = normal; 6 = fixed defect; 7 = reversable defect')
    
    #Converting variables to integers explicitly
    
    
    #Code for prediction
    Diagnosis = ' '
    
    #Creating a button for prediction
    if st.button('Heart disease Test Result'):
        Diagnosis = heartdisease_prediction([Age,Sex,Cp,Trestbps,Chol,Fbs,Restecg,Thalach,Exang,Oldpeak,Slope,Ca,Thal])
        
        
    st.success(Diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()



#streamlit run "D:\Deployment ML\HeartpredictionApp.py"






     
    