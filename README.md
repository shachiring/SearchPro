# E-commerce Visual Search

[Live Website](https://searchprod.netlify.app/)

Search visually similar products by uploading an image or using an image URL.

This repository contains:
- A FastAPI backend (`api.py`) for similarity search.
- A static frontend (`frontend/`) that calls the backend.
- An optional Streamlit app (`main.py`) for local experimentation.

## What It Does

- Loads product images from `db/images`.
- Builds/updates product metadata in `db/products.json`.
- Extracts image features:
  - Uses ResNet50 if TensorFlow is available.
  - Falls back to a lightweight color-histogram feature if not.
- Finds similar products using cosine similarity.

## Current Architecture

- Backend API: `api.py` (FastAPI + Uvicorn)
- Frontend UI: `frontend/index.html` + `frontend/app.js`
- Product catalog: `db/products.json`
- Product image storage: `db/images/`

## Project Structure

```text
.
|-- api.py
|-- main.py
|-- requirements.txt
|-- render.yaml
|-- netlify.toml
|-- db/
|   |-- products.json
|   `-- images/
`-- frontend/
    |-- index.html
    |-- app.js
    `-- styles.css
```

## Prerequisites

- Python 3.10+
- `pip`

## Local Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Backend (FastAPI)

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Useful URLs:
- API health: `http://localhost:8000/health`
- API docs (Swagger): `http://localhost:8000/docs`
- Debug info: `http://localhost:8000/debug`

## Run Frontend (Static UI)

From repo root:

```bash
python -m http.server 5173 --directory frontend
```

Open: `http://localhost:5173`

By default, the UI auto-detects backend URL:
- `http://localhost:8000` on localhost
- Render URL on Netlify

You can also set the backend URL manually in the UI.

## Optional: Run Streamlit App

`requirements.txt` is optimized for API deployment and does not include Streamlit.

If you want to run `main.py`:

```bash
pip install streamlit
streamlit run main.py
```

Optional (for deep features in Streamlit/API):

```bash
pip install tensorflow
```

Without TensorFlow, fallback features are used automatically.

## API Endpoints

### `GET /health`
Returns service status.

### `GET /debug`
Returns counts/shapes for loaded products and embeddings.

### `POST /search`
Find similar products from an uploaded image or URL.

Form fields:
- `file` (optional): image file
- `url` (optional): image URL
- `min_score` (optional, default `0.5`)
- `top_n` (optional, default `10`)

At least one of `file` or `url` is required.

Example with file:

```bash
curl -X POST "http://localhost:8000/search" \
  -F "file=@input/jeans1.jpg" \
  -F "min_score=0.4" \
  -F "top_n=5"
```

Example with URL:

```bash
curl -X POST "http://localhost:8000/search" \
  -F "url=https://example.com/sample.jpg" \
  -F "min_score=0.4" \
  -F "top_n=5"
```

## Catalog Behavior

- Real catalog images are read from `db/images`.
- If `db/products.json` is missing or contains placeholder-only items, the app rebuilds it from real images.
- Placeholder images (`product_000.png`, etc.) are ignored when real catalog images exist.

## Deployment

### Render (Backend)

`render.yaml` is configured to run:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Build command:

```bash
pip install -r requirements.txt
```

### Netlify (Frontend)

`netlify.toml` publishes `frontend/` and proxies API paths to the Render backend.

### Auto Deploy Trigger

GitHub Actions workflow `.github/workflows/deploy-render.yml` triggers Render deploys on push to `main`.

Required repository secrets:
- `RENDER_API_KEY`
- `RENDER_SERVICE_ID`

## Troubleshooting

- Build conflict between Pillow and Streamlit:
  - Keep `requirements.txt` for backend only.
  - Install Streamlit separately for local `main.py` usage.

- Search returns unrelated items:
  - Ensure real images are in `db/images`.
  - Restart backend so embeddings are rebuilt from current catalog.
  - Check `GET /debug` for `products_count` and `feature_vectors_shape`.

- Error: `No embeddings available`:
  - Verify image files referenced in `db/products.json` exist.
  - Restart backend and inspect startup logs for failed image loads.

## License

MIT. See `LICENSE`.
