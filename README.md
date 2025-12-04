AI-Powered Multi-Crop Pesticide Recommendation System
=====================================================

### 1. Project Overview

This project is an **AI-powered decision support system for small and medium farmers** growing **tomato, capsicum (bell pepper), and brinjal (eggplant)**.

From a single leaf image, the system:

- Detects the **plant type** (tomato / capsicum / brinjal).
- Identifies the **disease or pest** on the leaf.
- Estimates **severity level** (Low / Medium / High) using computer vision.
- Looks up a curated **pesticide recommendation database** with:
  - Recommended pesticide name,
  - Dose per acre,
  - Re‑entry interval (REI),
  - Maximum number of sprays,
  - Pre‑harvest interval (PHI),
  - Important safety/usage notes.
- Pulls **live weather data** (temperature, humidity, wind) from OpenWeather and adjusts the application advice.
- Computes the **exact dosage** required for the farmer’s plot size (acres or square meters).

The frontend presents all of this as a clean report: **Detected Disease, Severity, Recommended Pesticide, Best Time to Apply, Weather Conditions, REI, Max Sprays, PHI, and Dosage**.

The core goal is to **reduce overuse/misuse of pesticides**, save cost, and improve safety by giving farmers a simple, data‑backed recommendation instead of guesswork.

---

### 2. Tech Stack

- **Backend**
  - `Python 3`
  - **FastAPI** – REST API serving prediction and recommendation endpoint.
  - **TensorFlow / Keras** – Tomato disease model (`tomato_disease_model_v1.h5`).
  - **PyTorch** – Capsicum and eggplant disease models (`*.pth`).
  - **Hugging Face Transformers** – Plant classifier (`mobilevit-small` backbone).
  - **OpenCV / CV utilities** – For severity estimation (`calculate_severity_cv`).
  - **Requests** – For OpenWeather API calls.
  - **Custom JSON knowledge base** – `backend/recommendation_db_pkb.json` containing per‑crop, per‑disease, per‑severity pesticide data.

- **Frontend**
  - Plain **HTML5 / CSS3** responsive UI (`frontend/index.html`, `style.css`).
  - Vanilla **JavaScript** (`frontend/main.js`) for:
    - Camera capture or file upload,
    - Geolocation (to get latitude/longitude),
    - Calling the FastAPI backend,
    - Rendering the result cards.

- **Dev / Packaging**
  - `uvicorn` – ASGI server for FastAPI.
  - `requirements.txt` – Python dependency list.
  - Git & GitHub for version control.

---

### 3. Problem Statement & Motivation

Farmers often face three practical problems:

1. **Disease identification is hard**  
   Many diseases look similar (spots, blights, wilts). Without expertise, farmers struggle to correctly identify them.

2. **Pesticide misuse and overuse**  
   - Wrong molecule for the actual disease,  
   - Overdosing per acre,  
   - Ignoring safety periods (PHI/REI) and environmental conditions (heat, rain, wind).

3. **Fragmented information**  
   Expert recommendations exist in PDFs, research articles, or agri‑university booklets, but not in a format that is easy to query in the field.

This project tries to solve these by:

- Automating **plant + disease detection** from an image.
- Encoding expert **pesticide recommendations** into a structured knowledge base (`recommendation_db_pkb.json`).
- Combining it with **real‑time weather** and **user plot size** to produce a **simple, actionable recommendation**.

---

### 4. Repository Structure

Key parts of the codebase:

- `main.py`  
  FastAPI application entrypoint:
  - Loads models and class mappings.
  - Exposes `/predict_recommendation` endpoint.
  - Runs classifier → validates plant type → runs disease model → severity CV → recommendation engine → calls weather API.

- `backend/model_utils.py`  
  Model loading helpers, prediction utilities, severity calculation, and constants.

- `backend/recommendation_db_pkb.json`  
  Pesticide recommendation database segmented by:
  - Plant (`"tomato"`, `"capsicum"`, `"eggplant"`),
  - Disease (`"Wilt Disease"`, `"Early Blight"`, etc.),
  - Treatment (`"organic"`, `"chemical"`),
  - Severity (`"low"`, `"medium"`, `"high"`).

- `frontend/index.html`  
  Single‑page UI with:
  - Plant selector,
  - Land size + unit input,
  - Upload / Camera toggle,
  - Result card showing model outputs and recommendation.

- `frontend/main.js`  
  Frontend logic handling:
  - Geolocation,
  - Camera capture or file upload,
  - `fetch` call to `/predict_recommendation`,
  - DOM updates for disease name, severity, pesticide, weather, REI, max sprays, PHI, dosage.

- `requirements.txt`  
  Python dependencies needed for the backend.

---

### 5. How to Run the Project Locally

#### 5.1. Clone the repository

From any folder on your machine:

```bash
git clone https://github.com/Sherwinj10/AI-Powered-Pesticide-Recommendation-System-for-Tomato-Capsicum-Brinjal.git
cd AI-Powered-Pesticide-Recommendation-System-for-Tomato-Capsicum-Brinjal
```

#### 5.2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
macOS - source venv/bin/activate      # macOS / Linux
Windows - venv\Scripts\activate       # Windows (PowerShell or CMD)
```

#### 5.3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure the model files under `backend/Models` are present (they are large and might be stored via Git LFS or separate download if needed).

#### 5.4. Set your OpenWeather API key

In `main.py` you will see:

```python
OPENWEATHER_API_KEY = "YOUR_API_KEY_HERE"
```

Replace it with your own key from OpenWeather, or load it from an environment variable (for production you should not hardcode it).

#### 5.5. Run the FastAPI backend

From the project root:

```bash
python main.py
```

You should see log lines like:

- `Uvicorn running on http://127.0.0.1:8000`
- `Recommendation DB loaded successfully.`

#### 5.6. Open the frontend

The backend serves the static frontend automatically using `StaticFiles`.  
Once the server is running, open your browser and go to:

```text
http://127.0.0.1:8000
```

Then:

1. Select **plant type** (Tomato / Capsicum / Brinjal).
2. Enter **land size** and select units (acres or square meters).
3. Upload a clear **leaf image** or use the **camera** option.
4. Click **“Analyze Plant Health”**.
5. Wait for the loader to finish; the results card will show:
   - Detected disease,
   - Severity,
   - Recommended pesticide and dosage for your plot,
   - Best time to apply (with weather‑aware advice),
   - REI, maximum sprays, and PHI.

---

### 6. Possible Future Improvements

Some natural next steps:

- Add support for **more crops and diseases** (extend `recommendation_db_pkb.json` and train more models).
- Add **multilingual UI** for farmers (e.g., Hindi, Kannada, etc.).
- Replace hardcoded API key with **`.env` configuration**.
- Add a **feedback loop** (farmers marking whether the recommendation helped) to continuously improve the knowledge base.

This README is meant to be friendly both for **developers** (clone, set up, run) and for reviewers who want to understand **what problem the project solves** and **how**. 