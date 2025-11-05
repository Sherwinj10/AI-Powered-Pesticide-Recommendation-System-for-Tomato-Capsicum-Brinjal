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
        weatherDetails: document.getElementById("weather-details") 
    };
    const fileInput = document.getElementById('leaf-image');
    
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
        
        // Optional: Add dynamic styling for severity
        const severityClass = data.severity_class || 'severity-moderate';
        resultSpans.severityLevel.className = `result-value ${severityClass}`;
    }
});