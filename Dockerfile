FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tft_analytics/ tft_analytics/
COPY server/ server/
COPY scripts/ scripts/
COPY static/ static/
COPY data/ data/

ENV STATIC_DIR=/app/static

EXPOSE 8080

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "server.app:app"]
