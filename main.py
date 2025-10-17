from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd


model = joblib.load("model.pk1")


app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species_mapping = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}

@app.get("/")
async def home():
    """
    Home endpoint to check if the server is running.
    """  
    return {"message": "Welcome to the Iris Species Predictor API"}

@app.post("/predict")
async def predict_species(input_data: IrisInput):
    """
    Predict endpoint to classify the Iris species.
    Accepts JSON input and returns the predicted species.
    """
    # # Converts input data to numpy array
    # input_array = np.array([[input_data.sepal_length,
    #                          input_data.sepal_width,
    #                          input_data.petal_length,
    #                          input_data.petal_width]])                              
    
    feature_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    input_df = pd.DataFrame([input_data.dict()], columns=feature_names)

    #Predict using the model
    prediction = model.predict(input_df)[0] 

    #Get the species name
    species = species_mapping.get(prediction, "Unknown")

    return {"prediction": species}