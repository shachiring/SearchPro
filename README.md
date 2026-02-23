# Image Search Engine

This project is an image search engine that allows users to upload an image and find similar images from a preloaded database. It uses the ResNet50 model for feature extraction and cosine similarity for finding similar images.

## Features

- **Image Upload**: Users can upload an image in JPG format.
 - **Image URL**: Users can paste an image URL to search as well.
- **Feature Extraction**: Extracts features from the uploaded image using a pre-trained ResNet50 model.
- **Similarity Search**: Finds similar images from the database using cosine similarity.
- **Adjustable Parameters**: Users can adjust the similarity threshold and the number of similar images to display.
- **Caching**: Utilizes Streamlit's caching for improved performance.

- **Product DB**: App auto-generates a minimal product database (50 placeholder products with images and metadata) if `db/products.json` is missing.

## Prerequisites

- Python 3.10
- pip (Python package installer)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/nasirovsh/ecommerce-visual-search.git
    cd ecommerce-visual-search
    ```

2. **Create a virtual environment**:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

   For better performance, install the Watchdog module:
      
      For macOS users:
      ```sh
      xcode-select --install
      pip install watchdog
      ```
      For Windows users:
      ```sh
      pip install watchdog
      ```

## Usage

1. **Prepare the image database**:
   - Place your database images in the `db/` directory.
   - Ensure all images are in JPG format.

2. **Run the Streamlit app**:
   
   - Run in terminal:
      ```sh
      streamlit run main.py
      ```
   
   - Run/Debug in PyCharm:
   
      Go to Run/Debug Configurations and add a new configuration for Streamlit Server. Set module name to `streamlit` and parameters to `run main.py`.
   
      Click on the Run button to start the Streamlit server.

   - Open the browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. **Use the app**:
   - Upload an image or paste an image URL.
   - Adjust parameters: Use the sliders to set the similarity threshold and the number of similar images to display.
   - Find similar images: Click the "Search" button to search for similar products in the database.

## Deployment (free options)

- Streamlit Community Cloud: connect this repository and deploy; Streamlit handles the runtime.
- Render.com: create a new web service using the `streamlit run main.py` start command; choose a free plan.

Note: The app generates placeholder images at first run and caches feature vectors; cold-start may take longer due to model loading.

## Render (exact start command)

When creating a Web Service on Render use the following settings:

- Environment: `python`
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run main.py --server.port $PORT --server.headless true --server.enableCORS false`

You can also include `render.yaml` (included) for automatic service configuration when connecting the repo.

## Auto-deploy from GitHub to Render

This repo includes a GitHub Actions workflow that triggers a Render deploy when you push to `main`.

Setup steps:

1. In your GitHub repository, go to Settings → Secrets and variables → Actions → New repository secret. Add the following secrets:
   - `RENDER_API_KEY` — your Render API key (create in Render dashboard: Account → API Keys).
   - `RENDER_SERVICE_ID` — the Render Service ID for your web service (find in the service URL or Render dashboard).

2. Push to `main`. The workflow `.github/workflows/deploy-render.yml` will call the Render Deploys API to trigger a deploy.

If you prefer Render's native GitHub integration (recommended), connect the repo in the Render dashboard and enable auto-deploys — then you do not need the GitHub Action.

## Project Structure

- `main.py`: The main script that runs the Streamlit app.
- `requirements.txt`: Lists the dependencies required for the project.
- `db/`: Directory containing the image database.
- `README.md`: This file, containing project documentation.
- `LICENSE`: The license file for the project.

## Dependencies

- `pillow == 10.3.0`: Python Imaging Library for opening, manipulating, and saving images.
- `tensorflow == 2.16.1`: Open-source machine learning framework used for the ResNet50 model.
- `streamlit == 1.37.1`: Framework for building interactive web applications.
- `scikit-learn == 1.5.1`: Machine learning library used for cosine similarity calculation.
- `certifi == 2024.7.4`: Provides Mozilla's carefully curated collection of Root Certificates.

## How It Works

1. The app loads a pre-trained ResNet50 model on startup.
2. When a user uploads an image, the app extracts features using the ResNet50 model.
3. These features are compared to the pre-extracted features of images in the database using cosine similarity.
4. The app displays the most similar images based on the user-defined threshold and number of results.

## Troubleshooting

- If you encounter memory issues, try reducing the number of images in your database or upgrading your hardware.
- Ensure your uploaded images are in JPG format and are not corrupted.
- If the app is slow, it might be due to the initial loading of the database. Subsequent runs should be faster due to caching.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- The ResNet50 model is provided by TensorFlow and was originally developed by Microsoft Research.
- Thanks to the Streamlit team for their excellent framework for building data applications.