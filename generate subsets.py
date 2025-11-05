import os
import shutil
import random

# Paths
src = "/Users/shreeganeshnayak/Github-projects/AI-Powered-Multi-Crop-Pesticide-Recommendation-System/data/vegetables/Eggplant"
dst_train = "/Users/shreeganeshnayak/Github-projects/AI-Powered-Multi-Crop-Pesticide-Recommendation-System/data/vegetables/train/Eggplant"
dst_val = "/Users/shreeganeshnayak/Github-projects/AI-Powered-Multi-Crop-Pesticide-Recommendation-System/data/vegetables/val/Eggplant"

train_samples = 300
val_samples = 75

for class_name in os.listdir(src):
    class_path = os.path.join(src, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    # Choose subset
    selected_train = images[:train_samples]
    selected_val = images[train_samples:train_samples+val_samples]

    # Create folders
    os.makedirs(os.path.join(dst_train, class_name), exist_ok=True)
    os.makedirs(os.path.join(dst_val, class_name), exist_ok=True)

    # Copy subset
    for img in selected_train:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(dst_train, class_name, img))
    for img in selected_val:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(dst_val, class_name, img))

print("✅ Subset dataset created successfully.")
