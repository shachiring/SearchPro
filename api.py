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
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
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

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

model = None
feature_vectors = None
products = []


def load_model():
    global model
    if model is None:
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model


def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert('RGB')


def extract_features_from_pil(img: Image.Image):
    try:
        img = img.resize((224, 224))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        preds = model.predict(arr)
        return preds.flatten()
    except Exception:
        return None


def compute_embeddings():
    global feature_vectors, products
    if not os.path.exists(PRODUCTS_JSON):
        raise RuntimeError('products.json missing in db/')
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
        feature_vectors = np.vstack(features)
        np.save(EMBEDDINGS_NPY, feature_vectors)
        with open(PRODUCTS_ORDER, 'w', encoding='utf-8') as f:
            json.dump(valid_products, f, indent=2)
    else:
        feature_vectors = np.array([])


@app.on_event("startup")
def startup_event():
    global feature_vectors, products
    load_model()
    # load existing embeddings if present
    if os.path.exists(EMBEDDINGS_NPY) and os.path.exists(PRODUCTS_ORDER):
        try:
            feature_vectors = np.load(EMBEDDINGS_NPY)
            with open(PRODUCTS_ORDER, 'r', encoding='utf-8') as f:
                products = json.load(f)
            return
        except Exception:
            pass
    # compute embeddings
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
