# PlanAid

A system for processing and analyzing regulatory planning documents using NLP and machine learning.

## Prerequisites

- Docker and Docker Compose
- Python 3.10+ (if running locally)
- .NET 8.0 SDK (if running locally)
- Node.js 18+ (if running locally)

## Quick Start with Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd planaid
   ```

2. Create a `.env` file in the root directory, ask Are about details.

3. If not added in commit, get the .pth file from shared drive and add it to the ner_service/src/models folder.
   

3. Build and start all services:
   ```bash
   docker compose up -d --build
   ```

4. Access the services:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5251
   - Python Service: http://localhost:8000
   - NER Service: http://localhost:8001

5. Stop all services:
   ```bash
   docker compose down
   ```

   To remove all data (volumes):
   ```bash
   docker compose down -v
   ```

## Running Services Individually with Docker

### Frontend
```bash
cd frontend
docker build -t planaid-frontend .
docker run -p 3000:3000 planaid-frontend
```

### Backend
```bash
cd backend
docker build -t planaid-backend .
docker run -p 5251:5251 planaid-backend
```

### Python Service
```bash
cd python_service
docker build -t planaid-python .
docker run -p 8000:8000 planaid-python
```

### NER Service
```bash
cd ner_service
docker build -t planaid-ner .
docker run -p 8001:8001 planaid-ner
```

## Local Development Setup

### Using Conda (Alternative to Docker)

1. Create and activate conda environment:
   ```bash
   conda create -n planaid python=3.10
   conda activate planaid
   ```

2. Install Python dependencies:
   ```bash
   # For Python Service
   cd python_service
   pip install -r requirements.txt

   # For NER Service
   cd ner_service
   pip install -r requirements.txt
   python -m spacy download nb_core_news_md
   ```

3. Install .NET dependencies:
   ```bash
   cd backend
   dotnet restore
   ```

4. Install Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```

5. Start services locally:
   ```bash
   # Terminal 1 - Frontend
   cd frontend
   npm start

   # Terminal 2 - Backend
   cd backend
   dotnet watch run

   # Terminal 3 - Python Service
   cd python_service
   uvicorn app.main:app --reload --port 8000

   # Terminal 4 - NER Service
   cd ner_service
   uvicorn main:app --reload --port 8001
   ```

## Development Notes

- The services use hot-reload in development mode
- Docker Compose mounts volumes for live code updates
- Logs are available via `docker compose logs [service_name]`
- Health checks are implemented for all services

## Troubleshooting

1. If services fail to start:
   ```bash
   docker compose logs [service_name]
   ```

2. To rebuild a specific service:
   ```bash
   docker compose build [service_name]
   docker compose up -d [service_name]
   ```

3. Common issues:
   - Port conflicts: Check if ports 3000, 5251, 8000, or 8001 are in use
   - Volume permissions: Ensure proper read/write permissions
   - Missing .env file: Verify environment variables

## Architecture

- Frontend: React.js application
- Backend: .NET 8.0 Web API
- Python Service: FastAPI service for document processing
- NER Service: FastAPI service for NER model inference
- Redis: Cache for document processing results