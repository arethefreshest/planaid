# Base stage
FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=DEBUG

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pydantic-settings>=2.0.0
RUN python -m spacy download nb_core_news_md

# Copy the rest of the application
COPY . .

# Create logs directory
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Development stage
FROM base AS development
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# Production stage
FROM base AS production

# Install gunicorn
RUN pip install --no-cache-dir gunicorn

# Expose port
EXPOSE 8001

# Run gunicorn with proper logging configuration
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8001", "--workers", "4", "--timeout", "120", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "debug"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
