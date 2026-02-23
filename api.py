import os
import io
import json
import requests
import numpy as np
from typing import Optional
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# Try to import TensorFlow/Keras; fall back to a lightweight feature extractor if unavailable
USE_TF = False
try:
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.preprocessing import image
    USE_TF = True
except Exception:
    ResNet50 = None
    preprocess_input = None
    image = None
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = os.path.dirname(__file__)
DB_DIR = os.path.join(APP_DIR, 'db')
IMAGES_DIR = os.path.join(DB_DIR, 'images')
PRODUCTS_JSON = os.path.join(DB_DIR, 'products.json')
EMBEDDINGS_NPY = os.path.join(DB_DIR, 'embeddings.npy')
PRODUCTS_ORDER = os.path.join(DB_DIR, 'products_order.json')

app = FastAPI(title="Visual Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure images directory exists before mounting
os.makedirs(IMAGES_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

model = None
feature_vectors = None
products = []


def load_model():
    global model
    if model is None:
        if USE_TF:
            model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        else:
            model = None
    return model


def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert('RGB')


def extract_features_from_pil(img: Image.Image):
    try:
        # If TensorFlow/Keras is available, use ResNet50 features
        if USE_TF and model is not None and image is not None:
            img = img.resize((224, 224))
            arr = image.img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            preds = model.predict(arr)
            return preds.flatten()
        # Fallback: use a normalized color histogram (lightweight, no TF required)
        img_small = img.resize((224, 224))
        arr = np.array(img_small)
        # 32 bins per channel -> 96-dim vector
        hist = []
        for ch in range(3):
            h, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 255))
            hist.extend(h)
        hist = np.array(hist).astype('float32')
        # L1-normalize
        s = hist.sum() + 1e-9
        hist = hist / s
        return hist
    except Exception:
        return None


def generate_placeholder_products():
    """Generate placeholder products with simple colored images."""
    global products
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (255, 128, 0), (128, 255, 0), (128, 0, 255)
    ]
    categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books']
    
    products = []
    for i in range(50):
        color = colors[i % len(colors)]
        category = categories[i % len(categories)]
        
        # Generate a simple colored image
        img = Image.new('RGB', (100, 100), color=color)
        img_path = os.path.join(IMAGES_DIR, f'product_{i:03d}.png')
        img.save(img_path)
        
        products.append({
            'id': f'prod_{i:03d}',
            'name': f'{category} Product {i+1}',
            'category': category,
            'price': 10 + (i * 5),
            'description': f'A nice {category.lower()} item #{i+1}',
            'image': f'db/images/product_{i:03d}.png'
        })
    
    # Save products.json
    with open(PRODUCTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2)
    
    return products


def compute_embeddings():
    global feature_vectors, products
    if not os.path.exists(PRODUCTS_JSON):
        generate_placeholder_products()
    with open(PRODUCTS_JSON, 'r', encoding='utf-8') as f:
        products = json.load(f)
    features = []
    valid_products = []
    for p in products:
        img_path = p.get('image')
        if not img_path:
            continue
        full = os.path.join(APP_DIR, img_path)
        try:
            pil = Image.open(full).convert('RGB')
            feat = extract_features_from_pil(pil)
            if feat is not None:
                features.append(feat)
                valid_products.append(p)
        except Exception:
            continue
    if features:
        # Stack into 2D array (works for both TF and histogram features)
        try:
            feature_vectors = np.vstack(features)
        except Exception:
            feature_vectors = np.array(features)
        np.save(EMBEDDINGS_NPY, feature_vectors)
        with open(PRODUCTS_ORDER, 'w', encoding='utf-8') as f:
            json.dump(valid_products, f, indent=2)
    else:
        feature_vectors = np.array([])


@app.on_event("startup")
def startup_event():
    global feature_vectors, products
    # Load TF model only if available; otherwise rely on histogram fallback
    if USE_TF:
        load_model()
    # Always recompute embeddings from products.json to ensure consistency
    compute_embeddings()


@app.get('/health')
def health():
    return {"status": "ok"}


@app.post('/search')
async def search(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None),
                 min_score: float = Form(0.5), top_n: int = Form(10)):
    global feature_vectors, products
    if feature_vectors is None or feature_vectors.size == 0:
        return {"results": []}

    img_bytes = None
    if file is not None:
        img_bytes = await file.read()
    elif url:
        try:
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            img_bytes = resp.content
        except Exception:
            return {"error": "Failed to download image from URL"}
    else:
        return {"error": "No image provided"}

    try:
        pil = pil_from_bytes(img_bytes)
        qfeat = extract_features_from_pil(pil)
        if qfeat is None:
            return {"error": "Failed to extract features from query image"}
        sims = cosine_similarity([qfeat], feature_vectors)[0]
        results = []
        for idx, score in enumerate(sims):
            if score >= min_score:
                p = products[idx].copy()
                p['score'] = float(score)
                # expose image URL
                p['image_url'] = f"/images/{os.path.basename(p['image'])}"
                results.append(p)
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}
