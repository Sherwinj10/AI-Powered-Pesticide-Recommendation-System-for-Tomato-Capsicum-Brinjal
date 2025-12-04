# backend/model_utils.py

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import os
import json
import cv2
import numpy as np
from transformers import AutoImageProcessor, MobileViTForImageClassification

# --- 1. DEFINE PATHS AND IMAGE SIZES ---
BASE_PATH = os.path.join("backend", "Models")

MODEL_PATHS = {
    # Classifier
    "classifier": os.path.join(BASE_PATH, "mobilevit-small-finetuned-vegetables"), 

    # Disease Models
    "tomato": os.path.join(BASE_PATH, "tomato_disease_model_v1.h5"),
    "capsicum": os.path.join(BASE_PATH, "capsicum_model_v2_pytorch.pth"),
    "eggplant": os.path.join(BASE_PATH, "brinjal_model_v2_2_pytorch.pth") # Use new .pth
}

CLASS_JSON_PATHS = {
    # Disease Model JSONs (Classifier handles its own)
    "tomato": os.path.join(BASE_PATH, "tomato_class_indices.json"),
    "capsicum": os.path.join(BASE_PATH, "capsicum_class_indices.json"),
    "eggplant": os.path.join(BASE_PATH, "brinjal_class_indices.json") # Use new .json
}

MODEL_IMG_SIZES = {
    # Classifier size (processor handles this, but good to have)
    "classifier": (256, 256), 

    # Disease Model Sizes
    "tomato": (256, 256),    
    "capsicum": (224, 224),  
    "eggplant": (224, 224)   # New PyTorch model uses 224
}

# --- 2. IMAGE PREPROCESSING ---
def preprocess_image(image_bytes, model_type, img_size): 
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    if model_type == 'tf':
        img = img.resize(img_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
        
    elif model_type == 'torch':
        pytorch_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size), # Use CenterCrop for consistency
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = pytorch_transform(img)
        img_batch = img_tensor.unsqueeze(0)
        return img_batch

# --- 3. MODEL AND CLASS LOADING ---
def load_all_models_and_classes():
    loaded_models = {}
    class_mappings = {}
    
    try:
        # --- Load Hugging Face Classifier ---
        print("Loading classifier...")
        classifier_path = MODEL_PATHS["classifier"]
        loaded_models["classifier_processor"] = AutoImageProcessor.from_pretrained(classifier_path)
        loaded_models["classifier_model"] = MobileViTForImageClassification.from_pretrained(classifier_path)
        loaded_models["classifier_model"].eval()
        print("Classifier loaded.")

        # --- Load Keras Models ---
        print("Loading disease models...")
        loaded_models["tomato"] = models.load_model(MODEL_PATHS["tomato"])
        
        # --- Load PyTorch Models ---

        map_location = torch.device('cpu')
        loaded_models["capsicum"] = torch.load(MODEL_PATHS["capsicum"],map_location=map_location, weights_only=False)
        loaded_models["capsicum"].eval()

        loaded_models["eggplant"] = torch.load(MODEL_PATHS["eggplant"],map_location=map_location, weights_only=False)
        loaded_models["eggplant"].eval()
        
        print("All disease models loaded successfully.")

        # --- Load JSON Class Maps ---
        for plant_type, json_path in CLASS_JSON_PATHS.items():
            try:
                with open(json_path, 'r') as f:
                    # Load and invert the map
                    index_to_name_map = {int(k): v for k, v in json.load(f).items()}
                    class_mappings[plant_type] = index_to_name_map
            except Exception as e:
                print(f"Error loading JSON for '{plant_type}': {e}")
                
        print("All class mappings loaded.")
        
    except Exception as e:
        print(f"Error loading models or classes: {e}")
        print("Please check all file paths.")

    return loaded_models, class_mappings

# --- 4. PREDICTION FUNCTIONS ---

def predict_plant_type(image_bytes, model, processor):
    """
    Runs prediction using the Hugging Face classifier.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_id = logits.argmax(dim=-1).item()
        
        predicted_label = model.config.id2label[predicted_id]
        return predicted_label
    except Exception as e:
        print(f"Error in classifier prediction: {e}")
        return None

def predict_disease(model, image_bytes, model_type, class_map, img_size):
    """
    Runs prediction on a single image for disease models.
    """
    processed_image = preprocess_image(image_bytes, model_type, img_size)
    
    if model_type == 'tf':
        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction, axis=1)[0]
        
    elif model_type == 'torch':
        with torch.no_grad():
            prediction = model(processed_image)
            predicted_index = torch.argmax(prediction, dim=1).item()
            
    predicted_class_name = class_map.get(
        predicted_index, 
        f"Unknown (Index: {predicted_index})"
    )
    
    return predicted_class_name

def calculate_severity_cv(image_bytes):
    """
    Uses OpenCV to calculate disease severity as a percentage.
    This is a "hack" and is less accurate than a trained AI model.
    It works by counting "diseased" (brown/yellow) pixels vs. "healthy" (green) pixels.
    """
    try:
        # --- 1. Load Image ---
        # Convert image bytes to a numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the image into OpenCV format
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # --- 2. Convert to HSV Color Space ---
        # HSV (Hue, Saturation, Value) is much better for color segmentation
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

        # --- 3. Define Color Ranges ---
        # Green range (healthy leaf tissue)
        # These values are a good starting point but may need tuning.
        green_lower = np.array([25, 50, 20])
        green_upper = np.array([85, 255, 255])

        # Diseased range (brown/yellow spots)
        disease_lower = np.array([10, 50, 50])
        disease_upper = np.array([25, 255, 255])

        # --- 4. Create Masks ---
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        disease_mask = cv2.inRange(hsv, disease_lower, disease_upper)

        # --- 5. Calculate Pixel Counts ---
        green_pixels = cv2.countNonZero(green_mask)
        diseased_pixels = cv2.countNonZero(disease_mask)
        
        # Total "leaf" area is the sum of healthy and diseased pixels
        total_leaf_pixels = green_pixels + diseased_pixels
        
        if total_leaf_pixels == 0:
            # Avoid division by zero if no leaf is detected
            return "Not Available"

        # --- 6. Calculate Severity Percentage ---
        severity_perc = (diseased_pixels / total_leaf_pixels) * 100

        # --- 7. Return Severity String ---
        if severity_perc < 10:
            return "Low"
        elif severity_perc < 35:
            return "Medium"
        else:
            return "High"

    except Exception as e:
        print(f"Error in CV severity calculation: {e}")
        return "Not Available" # Fallback