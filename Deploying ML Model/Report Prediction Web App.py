# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 21:41:39 2025

@author: SHAMBHAVI ROY
"""

import os

import numpy as np
import pickle
import  streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "trained_model.sav")
)

print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)
print("EXISTS =", os.path.exists(MODEL_PATH))

with open(MODEL_PATH, "rb") as file:
    loaded_model = pickle.load(file)
# #loading the trained model
# loaded_model = pickle.load(open("C:/Users/SHAMBHAVI ROY/OneDrive/Desktop/Student's Report/trained_model.sav",'rb'))

loaded_rf = loaded_model['rf_regressor']
loaded_clf = loaded_model['classifier']
loaded_sc = loaded_model['scaler_clf']
loaded_sr = loaded_model['scaler_rf']

#creating a function for prediction

def report_prediction(input_data):
    
    input_np = np.array(input_data).reshape(1, -1)



    # Score prediction
    input_rf_scaled = loaded_sr.transform(input_np)
    
    predicted_score_rf = loaded_rf.predict(input_rf_scaled)[0]
    
    # Risk prediction
    input_clf_scaled = loaded_sc.transform(input_np)
    
    risk = loaded_clf.predict(input_clf_scaled)[0]
    
    
    if risk == 0:
       risk_msg = "⚠️ Early Warning: Student is at risk."
    else:
       risk_msg = "✅ Student is on the safe side."

    return f"Predicted Score: {predicted_score_rf:.2f}\n{risk_msg}"
        
        
def main():
    
    #giving a title
    st.title('Report Prediction Web App')
    
    #getting the input_data from the user
    
    weekly_self_study_hours = st.text_input('Weekly self-study hours (0–40)')
    attendance_percentage = st.text_input('Attendance percentage (50–100)')
    class_participation = st.text_input('Participation in Class(0-10)')
    
    #code for score prediction
    Score = ''
    
   
    
    #creating a button for score prediction
    
    if st.button('Predict Score'):
        Score = report_prediction([float(weekly_self_study_hours),
                                        float(attendance_percentage),
                                        float(class_participation),
                                        ])
        
        st.success(Score)
    
        
if __name__ == '__main__':
     main()
