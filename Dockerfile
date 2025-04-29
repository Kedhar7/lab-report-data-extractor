# 1) Start from a slim Python image
FROM python:3.11-slim

# 2) Install system packages for OCR & PDF→image
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      poppler-utils \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

# 3) Create & switch to app dir
WORKDIR /app

# 4) Copy & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# 5) Copy your application code
COPY . .

# 6) Tell Docker what port to expose
#    (Render will set $PORT at runtime)
ENV PORT=8000
EXPOSE ${PORT}

# 7) Launch via Gunicorn+Uvicorn worker
ENTRYPOINT ["sh", "-c", "exec gunicorn main:app \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT} \
    --timeout 120"]
