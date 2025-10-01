from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import sys
import pandas as pd
import numpy as np

from complaint_classifier.Text_preprocess import TextPreprocessor


# setting TextPreprocessor in sys.modules to avoid import issues becuse  during model bulding textpreprocessor was in main.py
sys.modules['__main__'].TextPreprocessor = TextPreprocessor



load_dotenv()
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
    return f"Welcome to Banking Complaint Classfier API {os.getenv('FRONT')}"

    
from complaint_classifier.complaint import predict_complaint as cp    
@app.post(
    "/complaint/predict",
    description="""
This endpoint predicts the category of a complaint based on its text.
The input should be a JSON object with the following structure:

```json
{
    "complaint": "string",
    "model": 1  # or 2
}

  model2 classes = ['Banking and Payments', 'Credit Card', 'Credit Reporting', 'Debt collection', 'Loan', 'Mortgage']
  model1 classes = ["Banking & Payments", "Credit & Debt", "Loans & Mortgages"]

```
}
"""
)
async def predict(data: dict):
   return  await cp(data)