FROM node:20-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY --from=frontend /build/../app/static app/static/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
