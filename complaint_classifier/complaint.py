
import joblib
from pydantic import BaseModel, ValidationError,Field
import pandas as pd

import numpy as np

from res import apiresponse,apierror

def load(path):
    model = joblib.load(path)
    return model


model1 = load('complaint_classifier/model1.pkl')
model2 = load('complaint_classifier/model2.pkl')


async def predict_complaint(data):
    text=data['complaint']
    if(text.strip()==""):
        return apierror(422,"Complaint text is empty")
    model=data['model']
    if(model not in [1,2]):
        return apierror(422,"Model should be 1 or 2")
    if(model==1):
        prediction = model1.predict_proba([text]).tolist()
    elif(model==2):
        prediction = model2.predict_proba([text]).tolist()
    return apiresponse(200,prediction,"Prediction successful")

