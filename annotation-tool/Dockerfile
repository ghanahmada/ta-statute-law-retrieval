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
COPY data/ data/
COPY --from=frontend /app/static app/static/

ENV PAIRS_TSV=/app/data/pairs.tsv
ENV ANNOTATORS_TXT=/app/data/annotators.txt
ENV DATABASE_URL=sqlite:///./db/labels.sqlite

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
