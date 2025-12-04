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
    calculate_severity_cv,
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
    """
    Look up pesticide recommendation from recommendation_db_pkb.json.

    The JSON schema is:
      {
        "Wilt Disease": {
          "treatment": {
            "chemical": {
              "low": [ { ... } ],
              "medium": [ { ... } ],
              "high": [ { ... } ]
            },
            "organic": [ ... ]
          }
        },
        ...
      }
    This function:
      1) Finds the disease entry whose key matches disease_name.
      2) Picks the first chemical recommendation for the given severity
         (falling back to medium -> low if needed).
      3) Uses dose_per_acre & unit from the JSON to compute final dosage.
    """

    # Normalize severity (e.g., "High" -> "high")
    severity_key = (severity or "").lower()

    # Default values in case nothing matches
    pesticide = "Generic Pesticide (Consult Local Expert)"
    base_dosage_per_acre = 100
    unit = "ml"
    notes = f"No recommendation found for '{disease_name}'. Please consult a local agricultural expert."

    # --- 1. Find matching disease entry in DB ---
    disease_entry = None
    for db_key, value in RECOMMENDATION_DB.items():
        # Keys can be things like "Wilt Disease", "Late Blight", etc.
        if db_key.lower() == disease_name.lower() or db_key.lower() in disease_name.lower():
            disease_entry = value
            break

    if disease_entry:
        treatment = disease_entry.get("treatment", {})
        chemical = treatment.get("chemical", {})

        # --- 2. Choose severity bucket (chemical[severity][0]) ---
        # Prefer exact severity; fall back to medium then low.
        rec_list = chemical.get(severity_key)
        if not rec_list:
            rec_list = chemical.get("medium") or chemical.get("low") or chemical.get("high")

        if rec_list:
            rec = rec_list[0]  # Take the first recommendation in the list
            pesticide = rec.get("name", pesticide)
            base_dosage_per_acre = rec.get("dose_per_acre", base_dosage_per_acre)
            unit = rec.get("unit", unit)
            notes = rec.get("notes", notes)

    # --- 3. Calculate Dosage ---
    if land_unit == "sqm":
        land_in_acres = land_size * 0.000247105
    else:
        land_in_acres = land_size
    
    total_dosage = base_dosage_per_acre * land_in_acres
    dosage_text = f"{total_dosage:.1f} {unit} for your {land_size} {land_unit} plot"

    # --- 4. Weather Advice Logic ---
    app_advice = "Apply in early morning."
    temp = weather_data.get("temp")
    
    if temp: # Check if temp is not None
        if temp > 32:
            app_advice = f"ADVISORY: High heat ({temp}°C). Apply in late evening."
        elif temp < 10:
            app_advice = f"ADVISORY: Too cold ({temp}°C). Wait for warmer weather."
    
    if "rain" in (weather_data.get("description", "")):
         app_advice = "ADVISORY: Rain detected. Do not spray now."
         
    # --- 5. Append notes from the database ---
    if notes:
        # Use <br> for a line break in HTML, but alert() might not render it.
        # A pipe separator is safer.
        app_advice = f"{app_advice} | IMPORTANT: {notes}"
         
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
    plant_type: str = Form(...), # This is the "intended" plant
    land_size: float = Form(...),
    land_unit: str = Form(...) 
):
    
    # Read image bytes ONCE
    image_contents = await file.read()

    # --- (STAGE 1, 2, 3: Classifier & Validation) ---
    # ... (This logic is all the same as before) ...
    # ... (It runs the classifier and checks for mismatch) ...
    
    # (Copying the logic from your file for clarity)
    form_plant_type = plant_type
    if form_plant_type == "brinjal": form_plant_type = "eggplant"
    
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
                detail=f"Image Mismatch: You selected '{form_plant_type.title()}', but this looks like a different plant. Please upload a '{form_plant_type.title()}' image."
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
        disease_name = prediction_name.replace("_", " ").replace("-", " ").title()
        severity_level = "N/A" # "N/A" is our code for "Healthy"
    else:
        # --- THIS IS THE KEY CHANGE ---
        # 1. Calculate severity using our new CV function
        print("Running CV severity analysis...")
        severity_level = calculate_severity_cv(image_contents) # <-- CALLING YOUR NEW FUNCTION
        print(f"CV Severity: {severity_level}")

        # 2. Parse the disease name (this logic is from your file)
        if "__" in prediction_name: # Handles "Tomato__Early_blight"
            disease_name = prediction_name.split('__')[-1].replace("_", " ").title()
        elif "_" in prediction_name: # Handles "Capsicum_Bacterial_spot"
            disease_name = prediction_name.split('_', 1)[-1].replace("_", " ").title()
        elif "-" in prediction_name: # Handles "Insect-pest-disease"
             disease_name = prediction_name.replace("-", " ").title()
        else:
            disease_name = prediction_name.title() # Fallback

    # --- STAGE 6 & 7: Recommendation Engine & Weather (UPDATED) ---
    recommendation_dict = {}
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
        
        # --- PASS THE NEW SEVERITY ---
        recommendation_dict = get_recommendation_logic(
            disease_name, 
            severity_level, # <-- Pass the CV severity ("Low", "Medium", "High")
            response_weather,
            land_size,
            land_unit
        )

    # --- STAGE 8: Return Structured JSON (UPDATED) ---
    print("Sending response to frontend.")
    return {
        "disease_name": disease_name,
        "severity_level": severity_level, # This will now show "Low", "Medium", or "High"
        "severity_class": f"severity-{severity_level.lower()}",
        "pesticide_recommendation": recommendation_dict["pesticide_name"],
        "application_time": recommendation_dict["application_advice"],
        "dosage_amount": recommendation_dict["dosage_text"],
        "weather_info": response_weather
    }

# --- 7. Serve Your HTML/CSS/JS (MUST COME LAST) ---
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# --- 8. Run the Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)