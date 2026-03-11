import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import onnxruntime as ort

# ── App ──
app = FastAPI(
    title="RiceGuard — Rice Leaf Disease Detection API",
    description="Detects Bacterial Leaf Blight, Brown Spot, and Leaf Smut using ResNet50V2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──
IMG_SIZE = (224, 224)
CLASSES = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut"]
CLASS_INFO = {
    "Bacterial Leaf Blight": {
        "emoji": "🦠",
        "color": "#ef4444",
        "description": "Caused by Xanthomonas oryzae. Leaves show water-soaked lesions that turn yellow then white.",
        "treatment": "Use resistant varieties, apply copper-based bactericides, ensure proper field drainage.",
        "severity": "High"
    },
    "Brown Spot": {
        "emoji": "🟤",
        "color": "#f97316",
        "description": "Caused by Cochliobolus miyabeanus. Small brown oval spots on leaves, often indicating nutrient deficiency.",
        "treatment": "Apply fungicides (propiconazole), improve soil nutrition, balanced NPK fertilization.",
        "severity": "Medium"
    },
    "Leaf Smut": {
        "emoji": "⚫",
        "color": "#a3e635",
        "description": "Caused by Entyloma oryzae. Small, slightly raised black spots on leaf surfaces.",
        "treatment": "Use certified disease-free seeds, crop rotation, apply systemic fungicides.",
        "severity": "Low"
    }
}

# ── Load ONNX model on startup ──
session = None

@app.on_event("startup")
async def load_model():
    global session
    model_path = "rice_model.onnx"
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        return
    print("Loading ONNX model...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    # Warmup
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    session.run(None, {session.get_inputs()[0].name: dummy})
    print(f"✅ ONNX model loaded!")
    print(f"   Input : {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"   Output: {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_inference(img_array: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    return outputs[0][0]  # shape: (3,)


def generate_gradcam_onnx(img_array: np.ndarray, class_idx: int) -> str | None:
    """
    Approximate Grad-CAM using occlusion-based sensitivity map.
    Works without TensorFlow gradients.
    """
    try:
        patch_size = 32
        stride = 16
        h, w = IMG_SIZE
        sensitivity_map = np.zeros((h, w), dtype=np.float32)
        baseline_pred = run_inference(img_array)[class_idx]

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                occluded = img_array.copy()
                occluded[0, y:y+patch_size, x:x+patch_size, :] = 0.5  # grey patch
                occluded_pred = run_inference(occluded)[class_idx]
                drop = baseline_pred - occluded_pred
                sensitivity_map[y:y+patch_size, x:x+patch_size] += drop

        # Normalize
        sensitivity_map = np.maximum(sensitivity_map, 0)
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()

        # Overlay
        img_np = (img_array[0] * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * sensitivity_map), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(superimposed_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


# ── Endpoints ──

@app.get("/")
def root():
    return {
        "status": "online",
        "model": "ResNet50V2 (ONNX)",
        "classes": CLASSES,
        "accuracy": "88.89%",
        "roc_auc": "0.9676"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": session is not None}

@app.get("/stats")
def stats():
    return {
        "model": "ResNet50V2 Transfer Learning (ONNX)",
        "dataset_size": 119,
        "classes": CLASSES,
        "test_accuracy": "88.89%",
        "roc_auc": "0.9676",
        "macro_f1": "0.8821",
        "ensemble_roc_auc": "1.0000",
        "techniques": [
            "Transfer Learning (ImageNet weights)",
            "2-Phase Training (Freeze → Fine-tune)",
            "Data Augmentation (10 techniques)",
            "Grad-CAM Explainability",
            "Test Time Augmentation",
            "Weighted Ensemble"
        ]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = preprocess(img)

        # Predict
        predictions = run_inference(img_array)
        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100

        all_probs = {
            CLASSES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASSES))
        }

        # Grad-CAM
        gradcam_b64 = generate_gradcam_onnx(img_array, predicted_idx)

        # Original image as base64
        buffer = io.BytesIO()
        img.resize(IMG_SIZE).save(buffer, format="PNG")
        original_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "all_probabilities": all_probs,
            "class_info": CLASS_INFO[predicted_class],
            "gradcam_image": gradcam_b64,
            "original_image": original_b64,
            "model_stats": {
                "model": "ResNet50V2 (ONNX)",
                "test_accuracy": "88.89%",
                "roc_auc": "0.9676",
                "dataset": "119 rice leaf images"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")