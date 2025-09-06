from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
from heart_disease.ht import predict_heart_disease as hdp
app = FastAPI()
origins = [
    os.getenv("FRONT")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # allowed origins
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, PUT, DELETE etc.
    allow_headers=["*"],          # all headers
)

@app.get("/")
def welcome():
    return "Welcome to Heart Disease Prediction API"


@app.post(
    "/heart_disease/predict",
    description="""
This endpoint predicts heart disease based on patient features.
The input should be a JSON object with the following structure:

```json
{
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
}
```
}
"""
)
async def predict(data: dict):
   return hdp(data)  # call your heart disease function
    

