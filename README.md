# Colorectal Histology Classification Web App

This repository contains a web application for colorectal histology image analysis using a saved TensorFlow/Keras CNN model. The app lets you upload an image, run tissue classification, view confidence scores, and inspect a Grad-CAM heatmap showing which regions influenced the prediction.

## Overview

The project is built from a trained colorectal histology classifier stored in [`model/colorectal_cancer_cnn.h5`](./model/colorectal_cancer_cnn.h5). A Flask backend handles image preprocessing, inference, and Grad-CAM generation, while the frontend provides a clean upload and results interface.

## Features

- Upload histopathology images from the browser
- Run prediction with the saved CNN model
- Display the predicted tissue class and confidence
- Show the top predicted classes with probability bars
- Generate a Grad-CAM overlay for visual explanation
- Run locally on `127.0.0.1` by default

## Supported Classes

The model predicts one of these eight tissue categories:

- `tumor`
- `stroma`
- `complex`
- `lympho`
- `debris`
- `mucosa`
- `adipose`
- `empty`

## Project Structure

```text
colorectal_cancer/
├── app.py
├── requirements.txt
├── LICENSE
├── README.md
├── model/
│   └── colorectal_cancer_cnn.h5
├── static/
│   ├── app.js
│   └── styles.css
├── templates/
│   └── index.html
├── outputs/
│   ├── confusion_matrix.png
│   ├── gradcam_comparison.png
│   ├── gradcam_result.png
│   ├── precision_recall_curves.png
│   ├── roc_curves.png
│   └── training_validation_accuracy.png
└── training_notebook/
    └── colorectal_cancer_April_1.ipynb
```

## Requirements

- Python `3.11`
- `pip` or Conda environment support
- TensorFlow-compatible local environment

## Installation

### Option 1: Conda environment

If you already have a Conda environment like `colon_cancer`, activate it first:

```bash
conda activate colon_cancer
pip install -r requirements.txt
```

### Option 2: Virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the App

Start the app with the default local port:

```bash
python app.py
```

This starts the server at:

```text
http://127.0.0.1:3000
```

You can choose a different port if needed:

```bash
python app.py --port 5000
python app.py --port 8000
```

## How to Use

1. Open the app in your browser.
2. Upload a colorectal histology image.
3. Click `Analyze Image`.
4. Review:
   - predicted class
   - confidence score
   - top predictions
   - Grad-CAM heatmap overlay

## Backend Notes

- The app loads the saved model from [`model/colorectal_cancer_cnn.h5`](./model/colorectal_cancer_cnn.h5).
- The expected input shape is inferred from the model and resolves to `150 x 150`.
- If direct Keras HDF5 loading fails because of serialization/version issues, the app rebuilds the CNN architecture and loads weights from the same saved model file.
- Grad-CAM is generated using the last convolutional layer and a separated classifier-head pass for stable gradient computation.

## API Endpoints

### `GET /`

Serves the web interface.

### `GET /health`

Returns a small health payload with:

- app status
- model filename
- input size
- last convolution layer name

### `POST /predict`

Accepts a multipart image upload and returns:

- predicted class
- confidence
- class description
- top predictions
- original image as base64
- Grad-CAM image as base64

## Troubleshooting

### App starts but prediction fails

- Make sure the uploaded file is a valid image.
- Confirm that `model/colorectal_cancer_cnn.h5` exists.
- Verify you installed dependencies in the active environment.

### Port already in use

Run the app on another port:

```bash
python app.py --port 5000
```

### TensorFlow or Keras loading issues

This project already includes a fallback loader in [`app.py`](./app.py) to handle common `.h5` compatibility issues across Keras versions.

## Important Disclaimer

This project is intended for educational, research, and demonstration purposes only. It is not a medical device and must not be used as a substitute for professional diagnosis, pathology review, or treatment planning.

## License

This project is licensed under the MIT License. See the [`LICENSE`](./LICENSE) file for details.


# final 14
