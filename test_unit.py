import pytest
import pandas as pd
from main import model, species_mapping


def test_model_prdicition():
    # Arrange: Create a test input DataFrame
    input_df = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

    # Act: Perform predicition
    prediction = model.predict(input_df)[0]
    species = species_mapping.get(prediction, "Unknown")

    # Assert: Check prediction result
    assert species == "Iris Setosa", "The predicition did not return the expected results"