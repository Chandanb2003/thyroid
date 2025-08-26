FROM python:3.9-slim

# Set environment variables for unbuffered output
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and torchvision directly with explicit URLs
RUN pip install --progress-bar on \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy requirements.txt for other packages
COPY requirements.txt .
RUN pip install --progress-bar on -r requirements.txt

# Copy application code
COPY . .

# Verify model exists
RUN test -f thyroid_model.pth || { echo "Model file missing!"; exit 1; }

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
