import torch
from torchvision import transforms
from PIL import Image
import os
from one_shot_model import Siamese  # Make sure this matches your model file

# ----------- CONFIGURATION -----------
MODEL_PATH = "siamese_model.pth"
REFERENCE_DIR = "training"  # or "training", must have subfolders for each class
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- PREPROCESSING -----------
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

# ----------- LOAD MODEL -----------
model = Siamese()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

def load_reference_images(reference_dir):
    """Load one reference image per class (or more if available)."""
    class_to_images = {}
    for class_name in os.listdir(reference_dir):
        class_path = os.path.join(reference_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = []
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                images.append(os.path.join(class_path, fname))
        if images:
            class_to_images[class_name] = images
    return class_to_images

def predict_class(input_image_path, class_to_images):
    """Compare input image to reference images from each class and predict."""
    img = Image.open(input_image_path).convert('L')
    img = transform(img).unsqueeze(0).to(DEVICE)

    scores = {}
    with torch.no_grad():
        for class_name, ref_paths in class_to_images.items():
            class_scores = []
            for ref_path in ref_paths:
                ref_img = Image.open(ref_path).convert('L')
                ref_img = transform(ref_img).unsqueeze(0).to(DEVICE)
                output = model(img, ref_img)
                prob = torch.sigmoid(output).item()
                class_scores.append(prob)
            scores[class_name] = sum(class_scores) / len(class_scores)
    # Pick class with highest average similarity
    predicted_class = max(scores, key=scores.get)
    return predicted_class, scores

if __name__ == "__main__":
    input_image_path = input("Enter Image path:").strip()
    if not os.path.isfile(input_image_path):
        print("Invalid image path.")
        exit(1)

    class_to_images = load_reference_images(REFERENCE_DIR)
    if not class_to_images:
        print(f"No reference images found in '{REFERENCE_DIR}'")
        exit(1)

    predicted_class, scores = predict_class(input_image_path, class_to_images)
    print("\n--- Prediction Results ---")
    for cls, score in scores.items():
        print(f"Class '{cls}': Similarity Score = {score:.3f}")
    print(f"\n✅ Predicted class: {predicted_class}")
