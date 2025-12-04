FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system deps (if you hit issues, we can extend this)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Hugging Face Spaces expects the app on port 7860
ENV PORT=7860

# Expose port
EXPOSE 7860

# Start FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]