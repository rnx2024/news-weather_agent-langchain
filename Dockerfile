FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY app ./app

# Uvicorn port is assigned by Render: $PORT
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
