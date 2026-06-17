# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 21:13:25 2025

@author: SHAMBHAVI ROY
"""

import numpy as np
import pickle
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "trained_model.sav")
)

import os
import pickle
import os

with open("trained_model.sav", "rb") as f:
    obj = pickle.load(f)

print(type(obj))
print(obj.keys())
print(os.path.getsize("trained_model.sav") / (1024**2), "MB")

print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)
print("MODEL EXISTS =", os.path.exists(MODEL_PATH))

root_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
print("ROOT DIR =", root_dir)
print("ROOT FILES =", os.listdir(root_dir))
print("CURRENT FILES =", os.listdir(BASE_DIR))

with open(MODEL_PATH, "rb") as file:
    loaded_model = pickle.load(file)
# #loading the trained model
# loaded_model = pickle.load(open("C:/Users/SHAMBHAVI ROY/OneDrive/Desktop/Student's Report/trained_model.sav",'rb'))

loaded_rf = loaded_model['rf_regressor']
loaded_clf = loaded_model['classifier']
loaded_sc = loaded_model['scaler_clf']
loaded_sr = loaded_model['scaler_rf']

input_data = (11.8,82.3,4.6)   #11,11.8,82.3,4.6,66.4,C,0

input_np = np.array(input_data).reshape(1, -1)



# Score prediction
input_rf_scaled = loaded_sr.transform(input_np)

predicted_score_rf = loaded_rf.predict(input_rf_scaled)[0]

# Risk prediction
input_clf_scaled = loaded_sc.transform(input_np)

risk = loaded_clf.predict(input_clf_scaled)[0]


print(f"Predicted Score: {predicted_score_rf:.2f}")

if risk == 0:
    print("⚠️Early Warning: Student is at risk.")
else:
    print("✅ Student is on the safe side.")