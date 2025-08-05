FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create directory for model weights
RUN mkdir -p utils/yolov9

# Download the yolov9-e-converted.pt file
RUN curl -L -o utils/yolov9/yolov9-e-converted.pt https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt

# Copy code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
