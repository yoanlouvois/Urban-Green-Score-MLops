FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY src/ ./src/
COPY entrypoint.py ./entrypoint.py

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "entrypoint.py"]