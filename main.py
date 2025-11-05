# main.py

import uvicorn
import requests
import io
import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from backend.model_utils import (
    load_all_models_and_classes, 
    predict_disease, 
    predict_plant_type,
    MODEL_IMG_SIZES 
)

# --- 2. Initialize App and Load Models/Classes (No Change) ---
app = FastAPI(title="AI Powered Pesticide Recommendation System for Tomato, Capsicum, and Brinjal")
MODELS, CLASS_MAPPINGS = load_all_models_and_classes()
if not MODELS or not CLASS_MAPPINGS:
    print("FATAL ERROR: Models or class mappings could not be loaded. Shutting down.")
    # In a real app, you might want to exit
    # exit()
DB_PATH = os.path.join("backend", "recommendation_db_ws.json")
try:
    with open(DB_PATH, 'r') as f:
        RECOMMENDATION_DB = json.load(f)
    print("Recommendation DB loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load recommendation_db.json: {e}")
    RECOMMENDATION_DB = {} # Start with an empty DB if load fails
# --- 3. API Key and Helper Functions (No Change) ---
OPENWEATHER_API_KEY = "347e7f996d0bf8ce6b23a60d5d6a076b"

def fetch_weather_data(lat, lon, api_key):
    """
    Calls OpenWeatherMap API to get current weather.
    """
    api_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"  # Gets temperature in Celsius
    }
    try:
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()  # Raises an error for bad responses
        data = response.json()
        
        # Extract only the data we need
        return {
            "temp": data.get('main', {}).get('temp'),
            "humidity": data.get('main', {}).get('humidity'),
            "wind_speed": data.get('wind', {}).get('speed'),  # meter/sec
            "description": data.get('weather', [{}])[0].get('description', 'N/A')
        }
    except requests.exceptions.RequestException as e:
        print(f"Weather API Error: {e}")
        # Return sensible defaults if API fails
        return {
            "temp": None, "humidity": None, "wind_speed": None, "description": "Weather data unavailable"
        }
def get_recommendation_logic(disease_name, severity, weather_data, land_size, land_unit):
    
    # --- 1. Look up the disease in our new database ---
    disease_info = RECOMMENDATION_DB.get(disease_name)
    severity_key = severity.lower()
    if not disease_info:
        # Fallback if disease is not in our JSON
        pesticide = "Generic Pesticide (Check Local Expert)"
        base_dosage_per_acre = 100 # Default
    else:
        # --- 2. Get recommendation based on severity ---
        # Normalize severity (e.g., "High" -> "high")
        severity_key = severity.lower() 
        
        if severity_key in disease_info:
            rec_data = disease_info[severity_key]
        else:
            # Default to 'medium' if severity is 'Unknown'
            rec_data = disease_info.get("medium", disease_info["low"]) 
            
        pesticide = rec_data.get("pesticide_name", "N/A")
        base_dosage_per_acre = rec_data.get("base_dosage_per_acre", 100)

    # --- 3. Calculate Dosage (Same as before) ---
    if land_unit == "sqm":
        land_in_acres = land_size * 0.000247105
    else:
        land_in_acres = land_size
    
    total_dosage = base_dosage_per_acre * land_in_acres
    dosage_text = f"{total_dosage:.1f} ml for your {land_size} {land_unit} plot" # Assuming ml

    # --- 4. Weather Advice Logic (Same as before) ---
    app_advice = "Apply in early morning."
    temp = weather_data.get("temp")
    
    if temp and temp > 32:
        app_advice = f"ADVISORY: High heat ({temp}°C). Apply in late evening."
    elif temp and temp < 10:
        app_advice = f"ADVISORY: Too cold ({temp}°C). Wait for warmer weather."

    if "rain" in (weather_data.get("description") or ""):
         app_advice = "ADVISORY: Rain detected. Do not spray now."
         
    return {
        "pesticide_name": pesticide,
        "application_advice": app_advice,
        "dosage_text": dosage_text
    }

# --- 4. Main Prediction Endpoint (Updated) ---
@app.post("/predict_recommendation")
async def predict_recommendation(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    plant_type: str = Form(...), # <-- We NOW use this as the "intended" plant
    land_size: float = Form(...),
    land_unit: str = Form(...) 
):
    
    # Read image bytes ONCE
    image_contents = await file.read()

    # --- STAGE 1: Get User's Intention ---
    # This is the plant the user *says* they are uploading (e.g., "tomato")
    form_plant_type = plant_type
    
    # Remap brinjal to eggplant for consistency
    if form_plant_type == "brinjal":
        form_plant_type = "eggplant"

    # --- STAGE 2: Run Classifier for Validation ---
    try:
        print("Running plant classifier for validation...")
        classifier_model = MODELS["classifier_model"]
        classifier_processor = MODELS["classifier_processor"]
        
        predicted_plant_type = predict_plant_type(
            image_contents,
            classifier_model,
            classifier_processor
        )
        if predicted_plant_type is None:
            raise Exception("Classifier returned None")
            
        print(f"Classifier prediction: {predicted_plant_type}")
        
    except Exception as e:
        print(f"Error during plant classification: {e}")
        raise HTTPException(status_code=500, detail=f"Plant classification failed: {e}")

    # --- STAGE 3: Compare and Validate ---
    
    # Normalize the classifier's prediction (e.g., "Bell_pepper" -> "capsicum")
    normalized_prediction = predicted_plant_type.lower().replace(" ", "_").replace("bell_pepper", "capsicum")
    if normalized_prediction == "brinjal":
        normalized_prediction = "eggplant"

    # This is your new logic!
    if normalized_prediction != form_plant_type:
        supported_plants = ["tomato", "capsicum", "eggplant"]
        
        # Case 1: The classifier predicted a *different* plant
        if normalized_prediction in supported_plants:
            raise HTTPException(
                status_code=400, # 400 = Bad Request
                detail=f"Image Mismatch: You selected '{form_plant_type.title()}', but this looks like a '{predicted_plant_type}' plant. Please upload a '{form_plant_type.title()}' image."
            )
        # Case 2: The classifier predicted something else (e.g., "None", "potato")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Image Not Recognized: This does not look like a plant. Please upload a '{form_plant_type.title()}' image."
            )

    # --- STAGE 4: Disease & Severity Model ---
    # If we get here, the check passed!
    # We now trust the form_plant_type and use it.
    
    print(f"Validation passed. Running disease model for: {form_plant_type}")
    
    if form_plant_type not in MODELS:
        # This check is still good, but it will use the validated type
        raise HTTPException(status_code=400, detail=f"Plant type '{form_plant_type}' is not supported.")
    if form_plant_type not in CLASS_MAPPINGS:
        raise HTTPException(status_code=400, detail=f"No class map for plant type: {form_plant_type}")
    if form_plant_type not in MODEL_IMG_SIZES: 
        raise HTTPException(status_code=400, detail=f"No image size for plant type: {form_plant_type}")
            
    model = MODELS[form_plant_type]
    class_map = CLASS_MAPPINGS[form_plant_type]
    img_size = MODEL_IMG_SIZES[form_plant_type]
    
    model_type = 'torch' if form_plant_type in ['capsicum', 'eggplant'] else 'tf'

    try:
        # Run disease prediction
        prediction_name = predict_disease(
            model, 
            image_contents, 
            model_type, 
            class_map,
            img_size
        )
        print(f"Disease model prediction: {prediction_name}")
        
    except Exception as e:
        print(f"Error during disease prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {e}")

   # --- STAGE 3: Parse Prediction ---
    if "healthy" in prediction_name.lower() or "health" in prediction_name.lower():
        # Correctly format the name (e.g., "Capsicum Healthy")
        disease_name = prediction_name.replace("_", " ").replace("-", " ").title()
        severity_level = "N/A" # "N/A" is our code for "Healthy"
    else:
        # Since no model predicts severity, set it to Not Available
        severity_level = "Not Available"

        # This complex parser finds the cleanest disease name
        if "__" in prediction_name: # Handles "Tomato__Early_blight"
            disease_name = prediction_name.split('__')[-1].replace("_", " ").title()
        elif "_" in prediction_name: # Handles "Capsicum_Bacterial_spot"
            # Split at the first underscore and take the rest
            disease_name = prediction_name.split('_', 1)[-1].replace("_", " ").title()
        elif "-" in prediction_name: # Handles "Insect-pest-disease"
             disease_name = prediction_name.replace("-", " ").title()
        else:
            disease_name = prediction_name.title()

    # --- STAGE 4 & 5: Recommendation Engine & Weather (LOGIC FIXED) ---
    recommendation_dict = {}
    # Default weather to N/A
    response_weather = {"temp": "N/A", "humidity": "N/A", "wind_speed": "N/A", "description": "N/A"}

    if severity_level == "N/A":
       recommendation_dict = {
            "pesticide_name": "None",
            "application_advice": "No action needed. Plant is healthy.",
            "dosage_text": "0 ml"
        }
    else:
        # ONLY fetch weather if the plant is sick
        print(f"Fetching weather for lat={lat}, lon={lon}")
        response_weather = fetch_weather_data(lat, lon, OPENWEATHER_API_KEY)
        
        recommendation_dict = get_recommendation_logic(
            disease_name, 
            severity_level,
            response_weather,
            land_size,
            land_unit
        )

    # --- STAGE 6: Return Structured JSON ---
    print("Sending response to frontend.")
    return {
        "disease_name": disease_name,
        "severity_level": severity_level,
        "severity_class": f"severity-{severity_level.lower()}",
        "pesticide_recommendation": recommendation_dict["pesticide_name"],
        "application_time": recommendation_dict["application_advice"],
        "dosage_amount": recommendation_dict["dosage_text"],
        "weather_info": response_weather # Send real or N/A weather
    }

# --- 7. Serve Your HTML/CSS/JS (MUST COME LAST) ---
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# --- 8. Run the Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)