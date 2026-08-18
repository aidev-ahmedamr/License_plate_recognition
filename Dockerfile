# Plate Reader API — containerized deployment
FROM python:3.11-slim

# System deps needed by opencv + easyocr's torch backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py api.py ./
# Drop your fine-tuned weights in alongside this Dockerfile and uncomment:
# COPY best.pt .
# ENV PLATE_MODEL_PATH=/app/best.pt

EXPOSE 5000
CMD ["python", "api.py"]
