FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed by OpenCV, WeasyPrint, etc.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Download models from Hugging Face Hub at build time
RUN python download_models.py

# HF Spaces requires the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
