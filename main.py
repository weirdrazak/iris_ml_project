from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd


model = joblib.load("model.pk1")


app = FastAPI()

# Mount static folder for serving static files like CSS and HTML
app.mount("/static", StaticFiles(directory="static"), name="static")



class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species_mapping = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}

@app.get("/", response_class=HTMLResponse)
async def home():
    # Serve the home page
    with open("static/index.html", "r") as file:
        return file.read()
    
@app.get("/predict", response_class=HTMLResponse)
async def predict():
    # Serve the prediction page
    with open("static/predict.html", "r") as file:
        return file.read()

@app.post("/predict")
async def predict_species(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)):
        
    feature_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)

    #Predict using the model
    prediction = model.predict(input_df)[0] 
    

    #Get the species name
    species = species_mapping.get(prediction, "Unknown")

    return {"prediction": species}