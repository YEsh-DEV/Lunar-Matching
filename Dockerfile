FROM python:3.12-slim

# System deps: gdal for rasterio, minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/jobs

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
