from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

#Import the local LLM service we just created
from app.llm_service import get_ai_insight

app = FastAPI(title="Freight Shipping Optimization API", version="1.0")

#lobal Variables for Models
regressor = None
classifier = None
model_columns = None

#Pydantic Model for Input Validation
class ShipmentRequest(BaseModel):
    origin: str
    destination: str
    distance: float
    mode: str
    weather: str
    traffic: str

#Startup Event: Load Models Once
@app.on_event("startup")
def load_models():
    global regressor, classifier, model_columns
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        regressor = joblib.load(os.path.join(base_dir, 'models', 'eta_regressor.pkl'))
        classifier = joblib.load(os.path.join(base_dir, 'models', 'risk_classifier.pkl'))
        model_columns = joblib.load(os.path.join(base_dir, 'models', 'model_columns.pkl'))
        print("Machine Learning Models loaded successfully.")
    except Exception as e:
        print(f" Error loading models: {e}")
        print("Please ensure you have run Notebook 03 to generate the .pkl files.")


def preprocess_input(data: ShipmentRequest) -> pd.DataFrame:
    #Create a base dictionary from the input
    input_dict = data.dict()
    
    #Add 'Is_Peak_Hour' (Rajesh didn't include time in the API req, so we default to current time)
    current_hour = datetime.now().hour
    is_peak = 1 if (8 <= current_hour <= 11) or (17 <= current_hour <= 20) else 0
    input_dict['Is_Peak_Hour'] = is_peak
    
    #Add 'Distance_Bucket'
    if data.distance <= 500:
        dist_bucket = 'Short'
    elif data.distance <= 1500:
        dist_bucket = 'Medium'
    else:
        dist_bucket = 'Long'
    
    #Convert to DataFrame
    df = pd.DataFrame([input_dict])
    
    #One-Hot Encode (Simulate the get_dummies process)
    for col in ['origin', 'destination', 'mode', 'weather', 'traffic']:
        val = df.at[0, col]
        dummy_col_name = f"{col.capitalize()}_{val}"
        df[dummy_col_name] = 1
        
    dummy_dist_name = f"Distance_Bucket_{dist_bucket}"
    df[dummy_dist_name] = 1
    
    #Drop original categorical columns
    df = df.drop(columns=['origin', 'destination', 'mode', 'weather', 'traffic'])
    
    #Align with Training Columns 
    df = df.reindex(columns=model_columns, fill_value=0)
    
    return df


@app.post("/predict-shipment")
async def predict_shipment(request: ShipmentRequest):
    if not regressor or not classifier:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    try:
        #Preprocess the incoming JSON
        processed_data = preprocess_input(request)
        
        #Predict Delay (Minutes)
        predicted_delay_mins = regressor.predict(processed_data)[0]
        
        #Calculate ETA (Hours)
        #Base speed assumptions: Truck=60km/h, Rail=40km/h, Flight=500km/h
        if request.mode == 'Truck':
            base_hours = request.distance / 60.0
        elif request.mode == 'Rail':
            base_hours = request.distance / 40.0
        else: # Flight
            base_hours = request.distance / 500.0
            
        eta_hours = base_hours + (predicted_delay_mins / 60.0)
        
        #Predict Risk and Confidence
        risk_label = classifier.predict(processed_data)[0]
        
        #Get probabilities to find the confidence score of the predicted class
        probabilities = classifier.predict_proba(processed_data)[0]
        predicted_class_index = list(classifier.classes_).index(risk_label)
        confidence = probabilities[predicted_class_index]
        
        #Get Local AI Insight (Gemma 2b via Ollama)
        ai_insight = get_ai_insight(
            shipment_data=request.dict(),
            eta_hours=eta_hours,
            risk_label=risk_label,
            confidence=confidence
        )
        
        #Return the final structured response
        return {
            "eta_hours": round(eta_hours, 2),
            "delay_risk": risk_label,
            "confidence": round(float(confidence), 2),
            "ai_insight": ai_insight
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#Root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "API is running. Send POST requests to /predict-shipment"}