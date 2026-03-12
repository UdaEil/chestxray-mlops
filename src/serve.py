import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os

# ── Config ───────────────────────────────────────────
import os
MODEL_PATH = os.getenv("MODEL_PATH", r"D:\chestxray-mlops\models\best_model.pt")
#MODEL_PATH = r"D:\chestxray-mlops\models/best_model.pt"
CLASSES    = ["NORMAL", "PNEUMONIA"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Model ───────────────────────────────────────
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

model = load_model()
print(f"✅ Model loaded from {MODEL_PATH}")

# ── Image Transform ──────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── FastAPI App ──────────────────────────────────────
app = FastAPI(
    title="Chest X-Ray Pneumonia Classifier",
    description="Upload a chest X-ray image to get a NORMAL / PNEUMONIA prediction",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Chest X-Ray Classifier API is running! Go to /docs to test it."}

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH, "device": str(DEVICE)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, predicted = probs.max(1)

    label      = CLASSES[predicted.item()]
    confidence = round(conf.item() * 100, 2)

    return JSONResponse({
        "prediction":  label,
        "confidence":  f"{confidence}%",
        "normal_prob":    f"{round(probs[0][0].item() * 100, 2)}%",
        "pneumonia_prob": f"{round(probs[0][1].item() * 100, 2)}%"
    })