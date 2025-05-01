# Base stage
FROM python:3.10-slim AS base
WORKDIR /app

# Install system dependencies including libmagic and pdfplumber
RUN apt-get update && apt-get install -y \
    libmagic1 \
    build-essential \
    python3-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY python_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base AS development
RUN pip install --no-cache-dir pytest pytest-asyncio
# Copy the entire python_service directory
COPY python_service/ .
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production
COPY python_service/ .
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose the port the app runs on
EXPOSE 8000