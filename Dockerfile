# API container (Cloud Run). The ML pipeline runs offline, not in this image.
FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY backend/ backend/
COPY mock_frontend/ mock_frontend/
ENV PORT=8080
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
