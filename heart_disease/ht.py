
from pydantic import BaseModel, ValidationError,Field
import pandas as pd

import numpy as np
import pickle

from res import apiresponse,apierror


def load():
    with open('heart_disease/model.pkl','rb') as f:
        model = pickle.load(f)
    return model

model = load()

class HeartData(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient's age in years (0-120)")
    sex: int = Field(..., ge=0, le=1, description="Sex (0 = Female, 1 = Male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=250, description="Resting blood pressure in mmHg")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0 = False, 1 = True)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (0 = No, 1 = Yes)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect, 3 = unknown)")

async def predict_heart_disease(data):
    try:
        hd_data = HeartData(**data)
    except ValidationError as e:
        return apierror(422,e.errors()[0]['msg'])
    df = pd.DataFrame([hd_data.model_dump()])
    prediction = model.predict_proba(df).tolist()
    return apiresponse(200,prediction,"Prediction successful")

