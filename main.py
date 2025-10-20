from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import joblib
import numpy as np
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("FastAPI Iris Predictor")

try:
    model = joblib.load("model.pk1")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Failed to load the model: %s", e)
    raise RuntimeError("Model loading failed.")

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
    try:
    # Serve the home page
        with open("static/index.html", "r") as file:
            logger.info("Home page served.")
            return file.read()
    except Exception as e:
        logger.error("Error serving home page: %s", e)
        return HTMLResponse(content="Error loading the home page.", status_code=500)
    
@app.get("/predict", response_class=HTMLResponse)
async def predict():
    try:
    # Serve the prediction page
        with open("static/predict.html", "r") as file:
            logger.info("Prediction page served.")
            return file.read()
    except Exception as e:
        logger.error("Error serving predict page: %s", e)
        return HTMLResponse(content="Error loading the prediction page.", status_code=500)
 


@app.post("/predict")
async def predict_species(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)):
        
    feature_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    try:
        input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)

        #Predict using the model
        
        prediction = model.predict(input_df)[0] 
        

        #Get the species name
        species = species_mapping.get(prediction, "Unknown")

        return {"prediction": species}
    except Exception as e:
        logger.error("Error making prediction: %s", e)
        return JSONResponse(content={"error": "Failed to make a prediction"}, status_code=500)

