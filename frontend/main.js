// main.js

// Wait for the document to be fully loaded

document.addEventListener("DOMContentLoaded", () => {
    
    // --- 1. Get All DOM Elements ---
    const analysisForm = document.getElementById("analysis-form");
    const loadingScreen = document.getElementById("loading-screen");
    const inputSection = document.getElementById("input-section");
    const resultSection = document.getElementById("result-section");

    // Get all the individual result spans
    const resultSpans = {
        diseaseName: document.getElementById("disease-name"),
        severityLevel: document.getElementById("severity-level"),
        pesticide: document.getElementById("pesticide-recommendation"),
        appTime: document.getElementById("application-time"),
        dosage: document.getElementById("dosage-amount"),
        // This ID is for the new weather field
        weatherDetails: document.getElementById("weather-details"),
        reiHours: document.getElementById("rei-hours"),
        maxSprays: document.getElementById("max-sprays"),
        phiDays: document.getElementById("phi-days"),
    };
    const fileInput = document.getElementById('leaf-image');
    const uploadModeBtn = document.getElementById("upload-mode-btn");
    const cameraModeBtn = document.getElementById("camera-mode-btn");
    const uploadInputWrapper = document.getElementById("upload-input");
    const cameraInputWrapper = document.getElementById("camera-input");
    const startCameraBtn = document.getElementById("start-camera");
    const capturePhotoBtn = document.getElementById("capture-photo");
    const retakePhotoBtn = document.getElementById("retake-photo");
    const cameraStreamEl = document.getElementById("camera-stream");
    const captureCanvas = document.getElementById("capture-canvas");
    const cameraPreview = document.getElementById("camera-preview");
    const capturedImageEl = document.getElementById("captured-image");
    
    let activeInputMode = "upload";
    let mediaStream = null;
    let capturedImageBlob = null;
    
    // Find the span element where the text should go
    // .closest() finds the nearest parent with this class
    const wrapper = fileInput.closest('.file-upload-wrapper'); 
    const fileTextSpan = wrapper.querySelector('.file-text');

    // Add an event listener for 'change'
    fileInput.addEventListener('change', function() {
      // Check if any file is selected
      if (fileInput.files.length > 0) {
        // Get the name of the first selected file
        const fileName = fileInput.files[0].name;
        
        // Update the span's text
        fileTextSpan.textContent = fileName;
      } else {
        // Reset to default if no file is selected
        fileTextSpan.textContent = 'Choose Image';
      }
    });

    uploadModeBtn.addEventListener("click", () => switchInputMode("upload"));
    cameraModeBtn.addEventListener("click", () => switchInputMode("camera"));
    startCameraBtn.addEventListener("click", startCamera);
    capturePhotoBtn.addEventListener("click", capturePhoto);
    retakePhotoBtn.addEventListener("click", resetCameraPreview);
    
    // --- 2. Add Form Submit Listener ---
    analysisForm.addEventListener("submit", (event) => {
        // Stop the default form browser submission
        event.preventDefault(); 
        
        // Show loading screen and hide form
        inputSection.classList.add("hidden");
        loadingScreen.classList.remove("hidden");
        
        // Start the analysis process (get location, then call API)
        getAnalysis();
    });

    // --- 3. Main Analysis Function ---
    async function getAnalysis() {
        try {
            // --- Step A: Get Geolocation ---
            const position = await getGeoLocation();
            const latitude = position.coords.latitude;
            const longitude = position.coords.longitude;

            // --- Step B: Get All Form Data ---
            // This automatically grabs all inputs from your form
            const formData = new FormData(analysisForm);
            handleImageAttachment(formData);
            
            // Add the location data to the FormData
            formData.append("lat", latitude);
            formData.append("lon", longitude);

            // --- Step C: Call the FastAPI Backend ---
            const response = await fetch("http://127.0.0.1:8000/predict_recommendation", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Analysis failed");
            }

            const data = await response.json();

            // --- Step D: Populate results and show them ---
            populateResults(data);
            
            loadingScreen.classList.add("hidden");
            resultSection.classList.remove("hidden");

        } catch (error) {
            // --- Error Handling ---
            console.error("Error:", error);
            alert("Error: " + error.message);
            
            // Reset UI on failure
            loadingScreen.classList.add("hidden");
            inputSection.classList.remove("hidden");
        }
    }

    // --- 4. Helper: Geolocation Promise ---
    function getGeoLocation() {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject(new Error("Geolocation is not supported by your browser."));
            } else {
                // This triggers the browser prompt
                navigator.geolocation.getCurrentPosition(resolve, () => {
                    reject(new Error("Unable to retrieve location. Please grant permission."));
                });
            }
        });
    }

    function switchInputMode(mode) {
        if (activeInputMode === mode) return;
        activeInputMode = mode;
        uploadModeBtn.classList.toggle("active", mode === "upload");
        cameraModeBtn.classList.toggle("active", mode === "camera");
        uploadInputWrapper.classList.toggle("hidden", mode !== "upload");
        cameraInputWrapper.classList.toggle("hidden", mode !== "camera");

        if (mode === "upload") {
            fileInput.required = true;
            stopCameraStream();
            capturedImageBlob = null;
            cameraPreview.classList.add("hidden");
        } else {
            fileInput.required = false;
            fileInput.value = "";
            fileTextSpan.textContent = "Choose Image";
            capturedImageBlob = null;
            cameraPreview.classList.add("hidden");
            startCamera();
        }
    }

    async function startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Camera access is not supported in this browser.");
            return;
        }
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            cameraStreamEl.srcObject = mediaStream;
            capturePhotoBtn.disabled = false;
        } catch (err) {
            console.error("Camera error:", err);
            alert("Unable to access camera. Please allow permission or use the upload option.");
        }
    }

    function stopCameraStream() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        cameraStreamEl.srcObject = null;
        capturePhotoBtn.disabled = true;
    }

    async function capturePhoto() {
        if (!mediaStream) {
            alert("Start the camera before capturing.");
            return;
        }

        const videoWidth = cameraStreamEl.videoWidth || 640;
        const videoHeight = cameraStreamEl.videoHeight || 480;
        captureCanvas.width = videoWidth;
        captureCanvas.height = videoHeight;

        const ctx = captureCanvas.getContext("2d");
        ctx.drawImage(cameraStreamEl, 0, 0, videoWidth, videoHeight);

        capturedImageBlob = await new Promise((resolve, reject) => {
            captureCanvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error("Failed to capture image"));
                }
            }, "image/jpeg", 0.9);
        });

        const previewUrl = URL.createObjectURL(capturedImageBlob);
        capturedImageEl.src = previewUrl;
        cameraPreview.classList.remove("hidden");
    }

    function resetCameraPreview() {
        capturedImageBlob = null;
        cameraPreview.classList.add("hidden");
    }

    function handleImageAttachment(formData) {
        if (activeInputMode === "camera") {
            if (!capturedImageBlob) {
                throw new Error("Please capture a photo before submitting.");
            }
            formData.delete("file");
            formData.append("file", capturedImageBlob, "camera_capture.jpg");
        } else {
            const file = fileInput.files[0];
            if (!file) {
                throw new Error("Please upload an image file.");
            }
        }
    }

    // --- 5. Helper: Populate Result Card ---
    function populateResults(data) {
        // data is the JSON object from FastAPI
        resultSpans.diseaseName.textContent = data.disease_name;
        resultSpans.severityLevel.textContent = data.severity_level;
        resultSpans.pesticide.textContent = data.pesticide_recommendation;
        resultSpans.appTime.textContent = data.application_time;
        resultSpans.dosage.textContent = data.dosage_amount;
        
        // This block formats and displays the REAL weather data
        const weather = data.weather_info;
        // This is the correct line
        if (weather && weather.temp !== null && typeof weather.temp === 'number') {
            // Convert wind speed from m/s to km/h
            const wind_kmh = (weather.wind_speed * 3.6).toFixed(1); 
            
            // Create a clean summary string
            resultSpans.weatherDetails.textContent = 
                `${weather.temp.toFixed(1)}°C, ${weather.humidity}% Humidity, ${wind_kmh} km/h Wind`;
        } else {
            resultSpans.weatherDetails.textContent = "Weather data unavailable";
        }

        // Safety and usage details from backend
        const rei = data.rei_hours;
        const maxSpray = data.max_sprays;
        const phi = data.phi_days;

        resultSpans.reiHours.textContent =
            rei !== null && rei !== undefined ? `${rei} hours` : "N/A";
        resultSpans.maxSprays.textContent =
            maxSpray !== null && maxSpray !== undefined ? `${maxSpray} per season` : "N/A";
        resultSpans.phiDays.textContent =
            phi !== null && phi !== undefined ? `${phi} days before harvest` : "N/A";
        
        // Optional: Add dynamic styling for severity
        const severityClass = data.severity_class || 'severity-moderate';
        resultSpans.severityLevel.className = `result-value ${severityClass}`;
    }
});