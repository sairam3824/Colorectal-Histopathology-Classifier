const STORAGE_KEY = "histologyPredictionHistory";
const MAX_HISTORY_ITEMS = 50;

function readPredictionHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    return [];
  }
}

function writePredictionHistory(items) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    return true;
  } catch (error) {
    return false;
  }
}

function savePredictionToHistory(payload, fileName) {
  const history = readPredictionHistory();
  const entry = {
    id: `${Date.now()}-${Math.floor(Math.random() * 100000)}`,
    createdAt: new Date().toISOString(),
    fileName: fileName || "uploaded-image",
    prediction: payload.prediction,
    topPredictions: payload.top_predictions,
    model: payload.model,
  };

  history.unshift(entry);
  const trimmed = history.slice(0, MAX_HISTORY_ITEMS);
  return writePredictionHistory(trimmed);
}

function formatTimestamp(isoString) {
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) {
    return "Unknown time";
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function renderHistoryPage() {
  const historyList = document.getElementById("history-list");
  const emptyState = document.getElementById("history-empty");
  if (!historyList || !emptyState) {
    return;
  }

  const entries = readPredictionHistory();
  historyList.innerHTML = "";

  if (entries.length === 0) {
    emptyState.classList.remove("hidden");
    return;
  }

  emptyState.classList.add("hidden");

  entries.forEach((entry) => {
    const card = document.createElement("article");
    card.className = "history-item";

    const header = document.createElement("div");
    header.className = "history-item-header";

    const title = document.createElement("h3");
    title.textContent = entry.prediction?.label || "Unknown class";

    const confidence = document.createElement("p");
    confidence.className = "history-confidence";
    const confidenceValue = Number(entry.prediction?.confidence || 0) * 100;
    confidence.textContent = `Confidence: ${confidenceValue.toFixed(2)}%`;

    const timestamp = document.createElement("p");
    timestamp.className = "history-timestamp";
    timestamp.textContent = formatTimestamp(entry.createdAt);

    header.appendChild(title);
    header.appendChild(confidence);
    header.appendChild(timestamp);

    const details = document.createElement("div");
    details.className = "history-details";

    const fileText = document.createElement("p");
    fileText.textContent = `File: ${entry.fileName || "uploaded-image"}`;

    const convText = document.createElement("p");
    convText.textContent = `Last Conv Layer: ${entry.model?.last_conv_layer || "n/a"}`;

    const descText = document.createElement("p");
    descText.textContent = entry.prediction?.description || "";

    details.appendChild(fileText);
    details.appendChild(convText);
    if (descText.textContent) {
      details.appendChild(descText);
    }

    const topWrapper = document.createElement("div");
    topWrapper.className = "history-top-predictions";

    const topTitle = document.createElement("p");
    topTitle.className = "history-top-title";
    topTitle.textContent = "Top Predictions";
    topWrapper.appendChild(topTitle);

    const topList = document.createElement("div");
    topList.className = "history-top-list";

    (entry.topPredictions || []).forEach((item) => {
      const chip = document.createElement("span");
      const value = Number(item.confidence || 0) * 100;
      chip.className = "history-chip";
      chip.textContent = `${item.label}: ${value.toFixed(1)}%`;
      topList.appendChild(chip);
    });

    topWrapper.appendChild(topList);

    card.appendChild(header);
    card.appendChild(details);
    card.appendChild(topWrapper);
    historyList.appendChild(card);
  });
}

function initializeHistoryPage() {
  const clearButton = document.getElementById("clear-history-button");
  if (!document.getElementById("history-list")) {
    return;
  }

  renderHistoryPage();

  if (clearButton) {
    clearButton.addEventListener("click", () => {
      writePredictionHistory([]);
      renderHistoryPage();
    });
  }
}

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

    savePredictionToHistory(payload, file.name);

    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    statusText.textContent = `Analysis complete. Model input size: ${payload.model.input_size.width} x ${payload.model.input_size.height}.`;
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    analyzeButton.disabled = false;
  }
}

if (
  form &&
  imageInput &&
  originalPreview &&
  resultOriginalImage &&
  gradcamImage &&
  analyzeButton &&
  statusText &&
  fileName &&
  resultsSection &&
  predictedLabel &&
  predictionDescription &&
  predictionConfidence &&
  lastConvLayer &&
  predictionBars &&
  dropzone
) {
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
}

initializeHistoryPage();
