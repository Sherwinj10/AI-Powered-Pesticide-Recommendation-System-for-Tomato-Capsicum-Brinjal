"""
Inference script for MobileViT vegetable classifier.
Loads a saved model, preprocesses an image, and returns JSON with the predicted label.
"""

import json
import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor, MobileViTForImageClassification
import torchvision.transforms as T

# Model path (change if needed)
MODEL_PATH = "/Users/shreeganeshnayak/Github-projects/AI-Powered-Multi-Crop-Pesticide-Recommendation-System/models/mobilevit-small-finetuned-vegetables"

# Device setup
device = "cpu"  # Use "cuda" for GPU, "mps" for Apple Silicon, or "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"


def load_model_and_processor(model_path: str):
    """Load the saved model and processor."""
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = MobileViTForImageClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


def preprocess_image(image_path: str, processor):
    """
    Preprocess image for inference.
    Uses the same preprocessing as validation (resize + center crop + normalize).
    """
    image = Image.open(image_path).convert("RGB")
    
    # Get image size from processor
    def _get_proc_attr(proc, name, default=None):
        if hasattr(proc, name):
            return getattr(proc, name)
        inner = getattr(proc, "image_processor", None)
        if inner is not None and hasattr(inner, name):
            return getattr(inner, name)
        return default
    
    size_cfg = _get_proc_attr(processor, "size", {})
    if isinstance(size_cfg, dict):
        if "height" in size_cfg and "width" in size_cfg:
            image_size = size_cfg["height"]
        elif "shortest_edge" in size_cfg:
            image_size = size_cfg["shortest_edge"]
        else:
            image_size = 256
    else:
        image_size = int(size_cfg) if size_cfg is not None else 256
    
    # Mean/std with fallback to ImageNet
    mean = _get_proc_attr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = _get_proc_attr(processor, "image_std", [0.229, 0.224, 0.225])
    
    # Validation transform (same as training)
    transform = T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    pixel_values = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return pixel_values


def predict(image_path: str, model=None, processor=None, model_path: str = MODEL_PATH):
    """
    Main inference function.
    Args:
        image_path: Path to the input image
        model: Pre-loaded model (optional, for batch inference)
        processor: Pre-loaded processor (optional, for batch inference)
        model_path: Path to the saved model directory (used if model/processor not provided)
    Returns:
        Predicted label string
    """
    # Load model and processor if not provided (for single inference)
    if model is None or processor is None:
        model, processor = load_model_and_processor(model_path)
    
    # Preprocess image
    pixel_values = preprocess_image(image_path, processor)
    
    # Run inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        predicted_id = logits.argmax(dim=-1).item()
    
    # Get label from model config
    id2label = model.config.id2label
    predicted_label = id2label[predicted_id]
    
    return predicted_label


def predict_json(image_path: str, model_path: str = MODEL_PATH):
    """
    Single image prediction that returns JSON string.
    """
    predicted_label = predict(image_path, model_path=model_path)
    result = {"label": predicted_label}
    return json.dumps(result, indent=2)


def batch_predict(test_dir: str, model_path: str = MODEL_PATH):
    """
    Run batch inference on all images in test directory.
    Args:
        test_dir: Path to test directory containing subfolders with images
        model_path: Path to the saved model directory
    Returns:
        List of dictionaries with "True Value" and "Predicted Value"
    """
    test_path = Path(test_dir)
    if not test_path.exists():
        raise ValueError(f"Test directory not found: {test_dir}")
    
    # Load model once for all predictions
    print(f"Loading model from {model_path}...")
    model, processor = load_model_and_processor(model_path)
    print("Model loaded successfully!")
    
    # Map folder names to model labels (handle case differences and variations)
    folder_to_label = {
        "bell_pepper": "Bell_pepper",
        "Bell_pepper": "Bell_pepper",
        "Bell_Pepper": "Bell_pepper",
        "eggplant": "Eggplant",
        "Eggplant": "Eggplant",
        "none": "None",
        "None": "None",
        "NONE": "None",
        "tomato": "Tomato",
        "Tomato": "Tomato",
        "TOMATO": "Tomato",
    }
    
    results = []
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".BMP"}
    
    # Get all subdirectories
    subdirs = [d for d in test_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(subdirs)} subdirectories in test folder")
    print("-" * 60)
    
    # Iterate through all subdirectories
    for folder_path in subdirs:
        folder_name = folder_path.name
        true_label = folder_to_label.get(folder_name, folder_name)
        
        # Process all images in this folder
        image_files = [
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in {ext.lower() for ext in image_extensions}
        ]
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in '{folder_name}' folder, skipping...")
            continue
        
        print(f"\n📁 Processing {len(image_files)} images from '{folder_name}' folder (True label: {true_label})...")
        
        processed = 0
        for image_path in image_files:
            try:
                predicted_label = predict(str(image_path), model=model, processor=processor)
                result = {
                    "True Value": true_label,
                    "Predicted Value": predicted_label
                }
                results.append(result)
                processed += 1
                if processed % 10 == 0:
                    print(f"  Processed {processed}/{len(image_files)} images...", end='\r')
            except Exception as e:
                print(f"\n  ❌ Error processing {image_path.name}: {e}")
                continue
        
        print(f"  ✅ Completed: {processed}/{len(image_files)} images processed")
    
    return results


def main():
    """CLI interface."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python classifier.py <image_path> [model_path]")
        print("  Batch inference: python classifier.py --batch <test_dir> [model_path]")
        sys.exit(1)
    
    # Check for batch mode
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Error: Test directory required for batch mode")
            print("Usage: python classifier.py --batch <test_dir> [model_path]")
            sys.exit(1)
        
        test_dir = sys.argv[2]
        model_path = sys.argv[3] if len(sys.argv) > 3 else MODEL_PATH
        
        if not Path(test_dir).exists():
            print(f"Error: Test directory not found: {test_dir}")
            sys.exit(1)
        
        if not Path(model_path).exists():
            print(f"Error: Model directory not found: {model_path}")
            sys.exit(1)
        
        # Run batch inference
        results = batch_predict(test_dir, model_path)
        
        # Output results in JSON format
        print("\n" + "="*60)
        print("BATCH INFERENCE RESULTS")
        print("="*60 + "\n")
        
        for result in results:
            print(json.dumps(result))
        
        # Calculate and print accuracy
        correct = sum(1 for r in results if r["True Value"] == r["Predicted Value"])
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print(f"SUMMARY: {correct}/{total} correct ({accuracy:.2f}%)")
        print("="*60)
        
    else:
        # Single image mode
        image_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
        
        image_path_obj = Path(image_path)
        
        if not image_path_obj.exists():
            print(f"Error: Path not found: {image_path}")
            sys.exit(1)
        
        # Check if it's a directory
        if image_path_obj.is_dir():
            print(f"Error: '{image_path}' is a directory, not an image file.")
            print("For batch inference on a directory, use: python classifier.py --batch <directory>")
            sys.exit(1)
        
        # Check if it's a file (not necessarily an image, but let PIL handle that)
        if not image_path_obj.is_file():
            print(f"Error: '{image_path}' is not a valid file.")
            sys.exit(1)
        
        if not Path(model_path).exists():
            print(f"Error: Model directory not found: {model_path}")
            sys.exit(1)
        
        result_json = predict_json(image_path, model_path)
        print(result_json)


if __name__ == "__main__":
    model, processor = load_model_and_processor(MODEL_PATH)
    processed_img = preprocess_image("/Users/shreeganeshnayak/Downloads/archive (2)/PlantVillage/Tomato_Leaf_Mold/0ced0bae-d224-43f5-8fd7-072c7cbd8f77___Crnl_L.Mold 9161.JPG")
    print(processed_img.shape)
    label = predict(processed_img)
    print(label)