import os
import io
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from one_shot_model import Siamese  # Make sure this is accessible

# ----------- CONFIGURATION -----------
# IMPORTANT: You need to have 'siamese_model.pth' and 'training' directory
# available in the same location as this FastAPI app when it runs.
# The 'training' directory should contain subfolders, each representing a class,
# with reference images inside.
MODEL_PATH = "siamese_model.pth"
REFERENCE_DIR = "training"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- PREPROCESSING -----------
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

# ----------- LOAD MODEL -----------
model = Siamese()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded successfully from {MODEL_PATH} on {DEVICE}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure it exists.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# ----------- LOAD REFERENCE IMAGES -----------
class_to_images = {}


def load_reference_images(reference_dir):
    """Load reference images from the specified directory."""
    global class_to_images  # Declare that we are modifying the global variable
    class_to_images = {}
    if not os.path.isdir(reference_dir):
        print(f"Warning: Reference directory '{reference_dir}' not found. No reference images will be loaded.")
        return

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

    if not class_to_images:
        print(f"Warning: No reference images found in '{reference_dir}'. Model will not be able to classify.")
    else:
        print(
            f"Loaded {sum(len(v) for v in class_to_images.values())} reference images for {len(class_to_images)} classes.")
        for cls, imgs in class_to_images.items():
            print(f"  Class '{cls}': {len(imgs)} images")


load_reference_images(REFERENCE_DIR)  # Initial load

# ----------- FASTAPI APP -----------
app = FastAPI(
    title="Siamese Network Inference API",
    description="API for classifying images using a pre-trained Siamese network.",
    version="1.0.0",
)


@app.get("/")
async def read_root():
    """Root endpoint for the API."""
    return {"message": "Welcome to the Siamese Network Inference API. Use /predict/image to classify an image."}


@app.post("/predict/image/")
async def predict_image(file: UploadFile = File(...)):
    """
    Predicts the class of an uploaded image using the Siamese network.

    Args:
        file (UploadFile): The image file to be classified.

    Returns:
        JSONResponse: A JSON object containing the predicted class and similarity scores.
    """
    if not class_to_images:
        raise HTTPException(
            status_code=503,
            detail="Reference images not loaded. Please ensure the 'training' directory is correctly set up."
        )

    try:
        # Read image from uploaded file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('L')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        scores = {}
        with torch.no_grad():
            for class_name, ref_paths in class_to_images.items():
                class_scores = []
                for ref_path in ref_paths:
                    ref_img = Image.open(ref_path).convert('L')
                    ref_img_tensor = transform(ref_img).unsqueeze(0).to(DEVICE)
                    output = model(img_tensor, ref_img_tensor)
                    prob = torch.sigmoid(output).item()
                    class_scores.append(prob)
                scores[class_name] = sum(class_scores) / len(class_scores)

        predicted_class = max(scores, key=scores.get)

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "similarity_scores": {cls: round(score, 4) for cls, score in scores.items()}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


@app.post("/reload_references/")
async def reload_references():
    """
    Reloads the reference images from the REFERENCE_DIR.
    Useful if you add new classes or images to the training directory
    without restarting the FastAPI application.
    """
    try:
        load_reference_images(REFERENCE_DIR)
        return JSONResponse(content={"message": "Reference images reloaded successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload reference images: {e}")

