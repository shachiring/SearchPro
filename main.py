import certifi
import os
import json
import random
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union
from io import BytesIO
import requests

# Constants
TRAINED_DB_PATH = "db"
PRODUCTS_JSON = os.path.join(TRAINED_DB_PATH, "products.json")
IMAGES_DIR = os.path.join(TRAINED_DB_PATH, "images")

# SSL Certificate setup
os.environ['SSL_CERT_FILE'] = certifi.where()


@st.cache_resource
def load_model() -> tf.keras.Model:
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')


def ensure_product_db(n_products: int = 50):
    """Create a simple product DB with generated images and metadata if missing."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    if os.path.exists(PRODUCTS_JSON):
        return

    categories = ["Shoes", "Bags", "Shirts", "Pants", "Hats"]
    products = []
    for i in range(1, n_products + 1):
        name = f"Product {i}"
        category = random.choice(categories)
        price = round(random.uniform(10, 200), 2)
        filename = f"product_{i:03d}.jpg"
        path = os.path.join(IMAGES_DIR, filename)
        create_placeholder_image(path, name, category)
        products.append({
            "id": i,
            "name": name,
            "category": category,
            "price": price,
            "image": os.path.relpath(path).replace('\\', '/')
        })

    with open(PRODUCTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2)


def create_placeholder_image(path: str, title: str, subtitle: str):
    """Generate a simple placeholder product image."""
    try:
        w, h = 400, 400
        img = Image.new('RGB', (w, h), color=tuple(random.choices(range(100, 256), k=3)))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        text = title
        subtitle = subtitle
        draw.text((20, 150), text, fill='black', font=font)
        draw.text((20, 190), subtitle, fill='black', font=font)
        img.save(path, format='JPEG', quality=85)
    except Exception:
        pass


def load_products():
    if not os.path.exists(PRODUCTS_JSON):
        ensure_product_db()
    with open(PRODUCTS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_feature_vectors_from_db(model: tf.keras.Model):
    products = load_products()
    feature_list = []
    valid_products = []
    for p in products:
        img_path = p.get('image')
        if not img_path:
            continue
        try:
            features = extract_features(img_path, model)
            if features is not None:
                feature_list.append(features)
                valid_products.append(p)
        except Exception:
            continue
    if not feature_list:
        return np.array([]), []
    feature_vectors = np.vstack(feature_list)
    return feature_vectors, valid_products


def extract_features(image_path: Union[str, BytesIO], model: tf.keras.Model) -> Union[np.ndarray, None]:
    try:
        if isinstance(image_path, BytesIO):
            img = image.load_img(image_path, target_size=(224, 224))
        else:
            img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img).flatten()
        return features
    except Exception:
        return None


def find_similar_products(query_image: Union[str, BytesIO], feature_vectors: np.ndarray, products: list[dict],
                          model: tf.keras.Model, min_score: float = 0.5, top_n: int = 10) -> list[dict]:
    query_features = extract_features(query_image, model)
    if query_features is None or feature_vectors.size == 0:
        return []
    similarities = cosine_similarity([query_features], feature_vectors)[0]
    results = []
    for idx, score in enumerate(similarities):
        if score >= min_score:
            p = products[idx].copy()
            p['score'] = float(score)
            results.append(p)
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
    return results


def load_image_from_url(url: str) -> Union[BytesIO, None]:
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        return BytesIO(resp.content)
    except Exception:
        return None


def main():
    st.set_page_config(page_title="E-commerce Visual Search", layout='centered')
    st.title("E-commerce Visual Search")
    st.markdown("Upload an image file or paste an image URL to find similar products.")

    ensure_product_db(50)
    model = load_model()

    with st.spinner("Preparing database and features..."):
        feature_vectors, products = get_feature_vectors_from_db(model)

    col1, col2 = st.columns([1, 2])
    with col1:
        upload = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        url = st.text_input("Or enter image URL")
        min_score = st.slider("Minimum similarity", 0.0, 1.0, 0.5, 0.01)
        top_n = st.slider("Max results", 1, 20, 10)
        search_btn = st.button("Search")

    query_img = None
    query_img_display = None
    if upload is not None:
        try:
            query_img = BytesIO(upload.read())
            query_img_display = Image.open(BytesIO(query_img.getvalue()))
        except Exception as e:
            st.error("Failed to read uploaded image.")

    if query_img is None and url:
        img_bytes = load_image_from_url(url)
        if img_bytes:
            query_img = img_bytes
            try:
                query_img_display = Image.open(BytesIO(img_bytes.getvalue()))
            except Exception:
                query_img_display = None
        else:
            st.error("Failed to download image from URL.")

    if query_img_display is not None:
        st.image(query_img_display, caption="Query image", use_column_width=True)

    if search_btn:
        if query_img is None:
            st.warning("Please upload an image or provide a valid image URL.")
        else:
            with st.spinner("Searching..."):
                results = find_similar_products(query_img, feature_vectors, products, model, min_score, top_n)
            if results:
                st.success(f"Found {len(results)} similar products")
                for r in results:
                    cols = st.columns([1, 3])
                    with cols[0]:
                        try:
                            img = Image.open(r['image'])
                            st.image(img, use_column_width=True)
                        except Exception:
                            st.write("[image missing]")
                    with cols[1]:
                        st.markdown(f"**{r['name']}**")
                        st.markdown(f"Category: {r['category']}")
                        st.markdown(f"Price: ${r['price']}")
                        st.markdown(f"Similarity: {r['score']:.3f}")
            else:
                st.info("No similar products found with the given minimum similarity.")


if __name__ == "__main__":
    main()