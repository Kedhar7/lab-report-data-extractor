# ---------- Dockerfile ----------

# 1) Base image
FROM python:3.11-slim

# 2) Install system deps for OCR (Tesseract), PDF → image (Poppler), OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tesseract-ocr \
      poppler-utils \
      libsm6 \
      libxext6 \
      libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# 3) Set working directory
WORKDIR /app

# 4) Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy your application code
COPY main.py utils.py ./

# 6) Expose port (optional; Render will inject $PORT)
EXPOSE 8000

# 7) Launch with Gunicorn + Uvicorn worker, binding to 0.0.0.0:$PORT (default 8000)
CMD ["sh", "-c", "gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000}"]
# ---------------------------------
