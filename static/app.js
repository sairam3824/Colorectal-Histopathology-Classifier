const form = document.getElementById("prediction-form");
const imageInput = document.getElementById("image-input");
const originalPreview = document.getElementById("original-preview");
const resultOriginalImage = document.getElementById("result-original-image");
const gradcamImage = document.getElementById("gradcam-image");
const analyzeButton = document.getElementById("analyze-button");
const statusText = document.getElementById("status-text");
const fileName = document.getElementById("file-name");
const resultsSection = document.getElementById("results-section");
const predictedLabel = document.getElementById("predicted-label");
const predictionDescription = document.getElementById("prediction-description");
const predictionConfidence = document.getElementById("prediction-confidence");
const lastConvLayer = document.getElementById("last-conv-layer");
const predictionBars = document.getElementById("prediction-bars");
const dropzone = document.getElementById("dropzone");

function setPreview(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    originalPreview.src = event.target.result;
    originalPreview.style.display = "block";
  };
  reader.readAsDataURL(file);
  fileName.textContent = file.name;
}

function renderPredictionBars(predictions) {
  predictionBars.innerHTML = "";

  predictions.forEach((item) => {
    const wrapper = document.createElement("div");
    wrapper.className = "prediction-bar";

    const header = document.createElement("div");
    header.className = "prediction-bar-header";
    header.innerHTML = `<span>${item.label}</span><span>${(item.confidence * 100).toFixed(1)}%</span>`;

    const track = document.createElement("div");
    track.className = "progress-track";

    const fill = document.createElement("div");
    fill.className = "progress-fill";
    fill.style.width = `${Math.max(item.confidence * 100, 2)}%`;

    track.appendChild(fill);
    wrapper.appendChild(header);
    wrapper.appendChild(track);
    predictionBars.appendChild(wrapper);
  });
}

async function submitPrediction(event) {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    statusText.textContent = "Please choose an image before running analysis.";
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  analyzeButton.disabled = true;
  statusText.textContent = "Running model inference and Grad-CAM generation...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed.");
    }

    predictedLabel.textContent = payload.prediction.label;
    predictionDescription.textContent = payload.prediction.description;
    predictionConfidence.textContent = `${(payload.prediction.confidence * 100).toFixed(2)}%`;
    lastConvLayer.textContent = payload.model.last_conv_layer;
    gradcamImage.src = payload.gradcam_image;
    gradcamImage.style.display = "block";
    resultOriginalImage.src = payload.original_image;
    resultOriginalImage.style.display = "block";
    renderPredictionBars(payload.top_predictions);

    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    statusText.textContent = `Analysis complete. Model input size: ${payload.model.input_size.width} x ${payload.model.input_size.height}.`;
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    analyzeButton.disabled = false;
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    return;
  }
  setPreview(file);
  statusText.textContent = "Image ready. Click analyze to run the model.";
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) {
    return;
  }
  imageInput.files = event.dataTransfer.files;
  setPreview(file);
  statusText.textContent = "Image ready. Click analyze to run the model.";
});

form.addEventListener("submit", submitPrediction);
// ver 2
// logic 9
