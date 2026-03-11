import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras

# ── App ──
app = FastAPI(
    title="Rice Leaf Disease Detection API",
    description="Detects Bacterial Leaf Blight, Brown Spot, and Leaf Smut from rice leaf images using ResNet50V2",
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

# ── Load model on startup ──
model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = "rice_disease_best_model.keras"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found at {model_path}")
        return
    print("Loading model...")
    model = keras.models.load_model(model_path)
    # Warmup
    dummy = np.zeros((1, *IMG_SIZE, 3))
    model.predict(dummy, verbose=0)
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")

# ── Health check ──
@app.get("/")
def root():
    return {
        "status": "online",
        "model": "ResNet50V2",
        "classes": CLASSES,
        "accuracy": "88.89%",
        "roc_auc": "0.9676"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

# ── Grad-CAM ──
def generate_gradcam(model, img_array, class_idx):
    try:
        # Find last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers'):  # base model
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        last_conv_layer = sublayer.name
                        break
                if last_conv_layer:
                    break
            elif isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break

        if not last_conv_layer:
            return None

        grad_model = keras.Model(
            model.inputs,
            [model.get_layer(last_conv_layer).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            class_channel = preds[:, class_idx]

        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Overlay on image
        img_np = (img_array[0] * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

        # Encode to base64
        pil_img = Image.fromarray(superimposed_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        gradcam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return gradcam_b64

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


# ── Predict endpoint ──
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and preprocess image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100

        # All class probabilities
        all_probs = {
            CLASSES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASSES))
        }

        # Grad-CAM
        gradcam_b64 = generate_gradcam(model, img_array, predicted_idx)

        # Original image as base64
        buffer = io.BytesIO()
        img_resized.save(buffer, format="PNG")
        original_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "all_probabilities": all_probs,
            "class_info": CLASS_INFO[predicted_class],
            "gradcam_image": gradcam_b64,
            "original_image": original_b64,
            "model_stats": {
                "model": "ResNet50V2",
                "test_accuracy": "88.89%",
                "roc_auc": "0.9676",
                "dataset": "119 rice leaf images"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── Stats endpoint ──
@app.get("/stats")
def stats():
    return {
        "model": "ResNet50V2 Transfer Learning",
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