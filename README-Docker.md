# Nurse Conversation Processing API - Docker Setup

> **Production-Ready Containerized Deployment Guide**

A scalable, containerized Python backend for processing nurse conversations with automatic transcription, Q&A extraction, and intelligent scoring using AI services.

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Docker Compose](https://img.shields.io/badge/docker--compose-2.20+-blue)](https://docs.docker.com/compose/)
[![Production Ready](https://img.shields.io/badge/production-ready-green)](https://github.com/your-repo)

This guide covers the complete Docker-based deployment of the Nurse Conversation Processing API, from development to production environments.

## 🏗️ System Architecture 

### High-Level Overview
The system is built as a microservices architecture with asynchronous task processing, featuring:

- **FastAPI Application Layer**: RESTful API with async processing
- **Celery Task Queue**: Distributed background job processing
- **AI Services Integration**: OpenAI Whisper + Google Gemini + OpenAI GPT
- **PostgreSQL Database**: Persistent data storage with async SQLAlchemy
- **Redis**: Message broker and result backend for Celery
- **Nginx**: Reverse proxy with SSL termination and load balancing

### Detailed Architecture Components

#### 1. **API Gateway & Application Layer**
- **FastAPI Application**: Port 8000, 4 Uvicorn workers with async processing
- **Middleware Stack**: CORS, authentication, request logging, error handling
- **API Endpoints**: File uploads, text processing, results retrieval, health checks
- **Data Validation**: Pydantic models with comprehensive request/response validation

#### 2. **Task Processing Infrastructure**
- **Celery Workers**: Multi-queue processing with dedicated transcription workers
- **Task Queues**: Priority-based routing (transcription, qa_extraction, processing, scoring)
- **Workflow Management**: Non-blocking task chaining with callback-based progression
- **Scheduler**: Celery Beat for periodic tasks and system maintenance

#### 3. **AI & Machine Learning Services**
- **Audio Transcription**: OpenAI Whisper (base model) with CPU optimization
- **Q&A Extraction**: Google Gemini API with fallback to OpenAI GPT
- **Intelligent Scoring**: Multi-dimensional conversation quality assessment
- **Model Management**: Version control, performance monitoring, fallback strategies

#### 4. **Data Processing Pipeline**
```
File Upload → Audio Transcription → Q&A Extraction → Scoring → Storage
     ↓              ↓                    ↓           ↓         ↓
  Validation    Whisper Model    Gemini/OpenAI   Algorithm  PostgreSQL
  & Security    Audio Processing  Context Analysis  Multi-Score  Async ORM
```

#### 5. **Database Schema**
- **conversation_uploads**: File metadata and processing status
- **conversation_transcriptions**: Audio-to-text results with timestamps
- **question_answers**: Extracted Q&A pairs with confidence scores
- **conversation_scores**: Multi-dimensional quality assessments
- **processing_jobs**: Task tracking and workflow management

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum | Recommended | Notes |
|-------------|---------|-------------|-------|
| **Docker** | 20.10+ | 24.0+ | Latest stable version |
| **Docker Compose** | 2.0+ | 2.20+ | V2 syntax support |
| **RAM** | 4GB | 8GB+ | More RAM = better performance |
| **Disk Space** | 10GB | 20GB+ | Includes models and data |
| **CPU** | 2 cores | 4+ cores | Multi-core for parallel processing |
| **Network** | Internet | Stable | For AI service APIs |

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd AmbientAI-BE

# Verify Docker installation
docker --version
docker-compose --version
```

### 2. Environment Configuration
```bash
# Copy environment template
cp env.example .env

# Edit configuration (required)
nano .env  # or your preferred editor

# Key settings to configure:
# - GEMINI_API_KEY (recommended for best results)
# - OPENAI_API_KEY (fallback service)  
# - SECRET_KEY (generate a secure random key)
# - Database passwords (change defaults)
```

**🔑 Important Environment Variables:**
```bash
# AI Service Keys (at least one recommended)
GEMINI_API_KEY=your_google_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Security (change in production!)
SECRET_KEY=your-super-secure-secret-key-here

# Database (change passwords!)
POSTGRES_PASSWORD=your_secure_password_here

# Performance tuning
CELERY_WORKER_CONCURRENCY=4
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
```

### 3. Start Production Environment
```bash
# Start all services in background
docker-compose up -d

# Monitor startup process
docker-compose logs -f

# Check all services are healthy
docker-compose ps
```

**Expected Output:**
```
NAME                        IMAGE                       STATUS
nurse-api                   nurse-conversation:latest   Up (healthy)
nurse-worker                nurse-conversation:latest   Up  
nurse-transcription-worker  nurse-conversation:latest   Up
nurse-beat                  nurse-conversation:latest   Up
nurse-flower               nurse-conversation:latest   Up (healthy)
nurse-postgres             postgres:15-alpine          Up (healthy)
nurse-redis                redis:7-alpine              Up (healthy)
nurse-nginx                nginx:alpine                Up
```

### 4. Verify Deployment
```bash
# Test API health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","timestamp":"2024-01-15T10:00:00Z","services":{"database":"healthy","celery":"healthy"}}

# Test API documentation
open http://localhost:8000/docs

# Test file upload (optional)
curl -X POST "http://localhost:8000/api/v1/uploads" \
  -F "file=@sample_audio.wav"
```

### 5. Access Services

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **API Documentation** | http://localhost:8000/docs | None | Interactive API docs |
| **Flower (Task Monitor)** | http://localhost:5555 | admin/flower123 | Celery task monitoring |
| **Health Check** | http://localhost:8000/health | None | System health status |
| **Nginx (optional)** | http://localhost:80 | None | Reverse proxy |
| **PostgreSQL** | localhost:5432 | postgres/postgres123 | Database direct access |
| **Redis** | localhost:6379 | None | Cache/broker access |

## 📦 Services Overview

| Service | Port | Description | Status |
|---------|------|-------------|---------|
| **api** | 8000 | FastAPI REST API server | ✅ Active |
| **worker** | - | General Celery workers | ✅ Active |
| **transcription-worker** | - | Dedicated audio workers | ✅ Active |
| **beat** | - | Celery beat scheduler | ✅ Active |
| **flower** | 5555 | Celery task monitoring | ✅ Active |
| **postgres** | 5432 | PostgreSQL database | ✅ Active |
| **redis** | 6379 | Redis message broker | ✅ Active |
| **nginx** | 80/443 | Reverse proxy (optional) | ⚠️ Configured |

## 🛠️ Management Commands

### Docker Compose Commands
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart specific service
docker-compose restart api

# View logs
docker-compose logs -f api          # API service logs
docker-compose logs -f worker       # Worker logs
docker-compose logs -f postgres     # Database logs

# Check service status
docker-compose ps

# Rebuild service (after code changes)
docker-compose build --no-cache api
docker-compose up -d api
```

### Database Operations
```bash
# Access PostgreSQL shell
docker-compose exec postgres psql -U postgres -d nurse_conversations

# Reset database (WARNING: deletes all data)
docker-compose down
docker volume rm nurseconversationextract_postgres_data
docker-compose up -d

# Backup database
docker-compose exec postgres pg_dump -U postgres nurse_conversations > backup.sql
```

### Development Commands
```bash
# Open shell in API container
docker-compose exec api bash

# Run tests
docker-compose exec api python -m pytest

# Check worker status
docker-compose exec api celery -A celery_app inspect active
```

## 🔧 Configuration

### Environment Variables (.env)

The system uses environment variables for all configuration. Here's a complete reference:

#### Database & Redis
```bash
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:postgres123@ambientai_be_postgres:5432/nurse_conversations

# Redis Configuration (Celery broker & result backend)
REDIS_URL=redis://ambientai_be_redis:6379/0
CELERY_BROKER_URL=redis://ambientai_be_redis:6379/0
CELERY_RESULT_BACKEND=redis://ambientai_be_redis:6379/0
```

#### AI Service API Keys
```bash
# Google Gemini (recommended for best Q&A results)
GEMINI_API_KEY=your_google_gemini_api_key_here

# OpenAI (fallback service)
OPENAI_API_KEY=your_openai_api_key_here

# Note: At least one AI service key is recommended
# System will work without keys but with reduced functionality
```

#### Whisper Transcription
```bash
# Model selection (affects accuracy vs speed)
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
WHISPER_DEVICE=cpu  # Options: cpu, cuda (if GPU available)

# Model Performance Comparison:
# tiny:   ~40MB, fastest, lower accuracy
# base:   ~150MB, good balance (recommended)
# small:  ~500MB, better accuracy
# medium: ~1.5GB, high accuracy
# large:  ~3GB, best accuracy
```

#### Application Settings
```bash
# Core application settings
SECRET_KEY=your-super-secure-secret-key-here-change-in-production
DEBUG=false  # Set to true for development
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
HOST=0.0.0.0
PORT=8000
WORKERS=4  # Number of API worker processes
```

#### File Upload Configuration
```bash
# Upload directory (inside container)
UPLOAD_DIR=/app/uploads

# File size limits
MAX_FILE_SIZE=104857600  # 100MB in bytes

# Supported audio formats (comma-separated)
ALLOWED_AUDIO_FORMATS=mp3,wav,m4a,ogg,flac
```

#### Celery Task Configuration
```bash
# Worker settings
CELERY_WORKER_CONCURRENCY=4  # Tasks per worker
CELERY_TASK_TIME_LIMIT=3600  # 1 hour max per task
CELERY_TASK_SOFT_TIME_LIMIT=3000  # 50 minutes soft limit

# Queue configuration (automatically handled)
CELERY_TASK_SERIALIZER=json
CELERY_RESULT_SERIALIZER=json
CELERY_ACCEPT_CONTENT=json
```

#### Q&A Model Settings
```bash
# Local Q&A model (fallback)
QA_MODEL=distilbert-base-cased-distilled-squad
MAX_QUESTION_LENGTH=512
MAX_CONTEXT_LENGTH=4096
CONFIDENCE_THRESHOLD=0.5
```

### Development vs Production Configuration

#### Development Environment
```bash
# Use docker-compose.dev.yml for development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Development-specific settings
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true  # FastAPI auto-reload
CELERY_WORKER_CONCURRENCY=1  # Easier debugging

# Additional dev services available:
# - Adminer (database admin): http://localhost:8080
# - Redis Commander: http://localhost:8081
# - Jupyter Notebooks: http://localhost:8888
```

#### Production Environment
```bash
# Production hardening checklist
DEBUG=false
LOG_LEVEL=WARNING
SECRET_KEY=your-very-long-random-secret-key-minimum-32-characters

# Use strong database passwords
POSTGRES_PASSWORD=your_very_secure_database_password

# External database (optional)
DATABASE_URL=postgresql+asyncpg://user:secure_pass@external-host:5432/db

# Performance optimization
CELERY_WORKER_CONCURRENCY=8  # Adjust based on CPU cores
WORKERS=4  # API worker processes

# Security headers and HTTPS
# Configure nginx.conf for SSL/TLS termination
```

### SSL/HTTPS Configuration

For production, configure SSL in `nginx.conf`:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    
    location / {
        proxy_pass http://ambientai_be_api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

## 📁 Directory Structure

```
├── app/                    # Application code
│   ├── main.py            # FastAPI application
│   ├── database.py        # Database connection
│   ├── database_models.py # SQLAlchemy models
│   ├── schemas.py         # Pydantic schemas
│   └── services/          # Business logic services
├── uploads/               # File uploads (mounted volume)
├── logs/                  # Application logs
├── notebooks/             # Jupyter notebooks (dev)
├── docker-compose.yml     # Production services
├── docker-compose.dev.yml # Development overrides
├── Dockerfile             # Production image
├── Dockerfile.dev         # Development image
├── nginx.conf             # Reverse proxy config
├── celery_app.py          # Celery configuration
├── tasks.py               # Celery task definitions
├── requirements.txt       # Python dependencies
└── init.sql              # Database initialization
```

## 🔍 API Usage Examples

### Upload Audio File
```bash
curl -X POST "http://localhost:8000/api/v1/uploads" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@conversation.wav"
```

### Process Text Only
```bash
curl -X POST "http://localhost:8000/api/v1/text-processing" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient complains of chest pain..."}'
```

### Get Processing Results
```bash
curl "http://localhost:8000/api/v1/uploads/{upload_id}"
```

### Check Task Status
```bash
curl "http://localhost:8000/api/v1/tasks/{task_id}"
```

### Response Format
```json
{
  "upload": {
    "id": "uuid-string",
    "filename": "conversation.wav",
    "status": "completed",
    "created_at": "2024-01-01T10:00:00Z"
  },
  "transcription": {
    "text": "Full conversation transcript...",
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "text": "Hello, I'm experiencing chest pain."
      }
    ]
  },
  "qa_results": [
    {
      "question_id": "chief_complaint",
      "question_text": "What is the patient's chief complaint?",
      "answer_text": "chest pain",
      "confidence_score": 0.95,
      "timestamp_start": 10.5,
      "timestamp_end": 12.3,
      "is_confident": true
    }
  ],
  "score": {
    "completeness_score": 0.85,
    "confidence_score": 0.92,
    "patient_info_score": 0.78
  }
}
```

## 🔒 Security Features

- **Container Security**: Non-root container users, minimal attack surface
- **API Security**: Rate limiting, input validation, SQL injection protection
- **Data Protection**: File upload restrictions, format validation, virus scanning
- **Authentication**: API key management, session handling
- **Encryption**: Data encryption at rest and in transit
- **Audit Logging**: Comprehensive access and change logging

## 📊 Monitoring & Observability

### Health Checks
```bash
# API health endpoint
curl http://localhost:8000/health

# Service health check
docker-compose ps
```

### Logging
- **Application Logs**: `./logs/app.log` (JSON format)
- **Container Logs**: `docker-compose logs -f [service]`
- **Structured Logging**: JSON format with correlation IDs

### Monitoring Tools
- **Flower**: Celery task monitoring (http://localhost:5555)
- **Database Metrics**: PostgreSQL statistics and performance
- **Redis Metrics**: Memory usage, connection counts, queue lengths
- **System Metrics**: CPU, memory, disk I/O, network performance

### Key Metrics to Monitor
- Task queue lengths and processing times
- API response times and error rates
- Database connection pool usage
- File upload success rates
- AI service API response times

## 🚀 Performance Tuning

### For High Load
```yaml
# In docker-compose.yml
api:
  deploy:
    replicas: 4  # Scale API instances

worker:
  deploy:
    replicas: 3  # Scale workers

# Environment variables
CELERY_WORKER_CONCURRENCY=8
CELERY_TASK_TIME_LIMIT=7200
```

### GPU Support (Optional)
```yaml
# Add to worker services for GPU acceleration
worker:
  runtime: nvidia
  environment:
    - WHISPER_DEVICE=cuda
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Database Optimization
```sql
-- Create performance indexes
CREATE INDEX idx_uploads_status ON conversation_uploads(status);
CREATE INDEX idx_uploads_created_at ON conversation_uploads(created_at);
CREATE INDEX idx_transcriptions_upload_id ON conversation_transcriptions(upload_id);
```

## 🛠️ Development

### Hot Reloading
Development environment includes:
- Code hot reloading for FastAPI
- Celery worker auto-restart
- Volume mounts for live editing
- Debug ports exposed

### Adding New Features
1. Modify code in the appropriate modules
2. Add tests for new functionality
3. Update schemas and models as needed
4. Rebuild and restart services
5. Test with the API endpoints

### Database Migrations
```bash
# After model changes, reset database
docker-compose down
docker volume rm nurseconversationextract_postgres_data
docker-compose up -d
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. **Services Won't Start**
```bash
# Check Docker daemon
docker info

# Check logs for specific errors
docker-compose logs api
docker-compose logs worker

# Reset everything
docker-compose down
docker volume prune -f
docker-compose up -d
```

#### 2. **API Not Responding**
```bash
# Check API container status
docker-compose ps api

# Check API logs
docker-compose logs -f api

# Verify database connection
docker-compose exec api python -c "from database import get_db; print('DB OK')"
```

#### 3. **Celery Workers Not Processing**
```bash
# Check worker status
docker-compose exec api celery -A celery_app inspect active

# Check Redis connection
docker-compose exec redis redis-cli ping

# Restart workers
docker-compose restart worker transcription-worker
```

#### 4. **Database Connection Errors**
```bash
# Check PostgreSQL health
docker-compose exec postgres pg_isready -U postgres

# Reset database (WARNING: deletes data)
docker-compose down
docker volume rm nurseconversationextract_postgres_data
docker-compose up -d
```

#### 5. **Out of Disk Space**
```bash
# Clean unused Docker resources
docker system prune -af
docker volume prune -f

# Check disk usage
docker system df
```

#### 6. **File Upload Issues**
```bash
# Check uploads directory permissions
docker-compose exec api ls -la /app/uploads

# Verify file size limits in .env
# Check allowed file extensions
```

### Performance Issues
- **Monitor resource usage**: `docker stats`
- **Check worker queues**: Flower UI (http://localhost:5555)
- **Review application logs**: Look for bottlenecks and errors
- **Database performance**: Check connection pool and query performance
- **Consider scaling**: Add more workers or API instances

### Debug Mode
```bash
# Enable debug logging
DEBUG=true
LOG_LEVEL=DEBUG

# Rebuild and restart
docker-compose build --no-cache api
docker-compose up -d api
```

## 📈 Recent System Improvements

### 1. **Celery Task Architecture Refactor**
- Replaced blocking `result.get()` calls with non-blocking workflow
- Implemented callback-based task progression
- Added dedicated task routing and queue management

### 2. **Database Interaction Fixes**
- Fixed SQLAlchemy lazy loading issues with `selectinload`
- Resolved datetime serialization problems
- Added comprehensive error handling for database operations

### 3. **AI Service Integration**
- Google Gemini API integration for Q&A extraction
- OpenAI GPT fallback service
- Whisper model optimization for CPU processing

### 4. **Error Handling & Resilience**
- Comprehensive exception handling with proper HTTP status codes
- Retry mechanisms for transient failures
- Circuit breaker patterns for external service calls

### 5. **Data Validation & Security**
- Extended Pydantic schemas with missing enum values
- Enhanced input validation and sanitization
- Improved file upload security and validation

## 👨‍💻 Author

**Soumyadeep** - *Lead Developer & System Architect*

- 🏗️ **System Architecture**: Designed scalable microservices architecture with Docker & Celery
- 🤖 **AI Integration**: Implemented hybrid AI pipeline with Gemini, OpenAI, and DistilBERT
- 📊 **Database Design**: Created comprehensive PostgreSQL schema for medical data
- 🐳 **DevOps**: Docker containerization, multi-environment deployment strategies
- 📚 **Documentation**: Comprehensive technical documentation and deployment guides

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test locally
4. Submit a pull request

*All contributions are welcome! Please reach out to **Soumyadeep** for major feature discussions.*

## 📞 Support

For issues and questions:
- Check the logs: `docker-compose logs -f [service]`
- Review health status: `docker-compose ps`
- Monitor tasks: Flower UI (http://localhost:5555)
- Contact **Soumyadeep** for technical support
- Open an issue on GitHub

## 🚢 Advanced Deployment Strategies

### Multi-Environment Setup

#### 1. Development Environment
```bash
# Full development stack with debugging tools
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Available services:
# - API with hot reload: http://localhost:8000
# - Flower monitoring: http://localhost:5555
# - Database admin: http://localhost:8080
# - Redis admin: http://localhost:8081
# - Jupyter notebooks: http://localhost:8888
```

#### 2. Staging Environment
```bash
# Create staging override file
cat > docker-compose.staging.yml << EOF
version: '3.8'
services:
  ambientai_be_api:
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - SECRET_KEY=staging-secret-key
    deploy:
      replicas: 2

  ambientai_be_worker:
    environment:
      - CELERY_WORKER_CONCURRENCY=2
    deploy:
      replicas: 2
EOF

# Deploy staging
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```

#### 3. Production Environment
```bash
# Production with scaling and security
cat > docker-compose.prod.yml << EOF
version: '3.8'
services:
  ambientai_be_api:
    environment:
      - DEBUG=false
      - LOG_LEVEL=WARNING
      - SECRET_KEY=${PRODUCTION_SECRET_KEY}
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G

  ambientai_be_worker:
    environment:
      - CELERY_WORKER_CONCURRENCY=8
    deploy:
      replicas: 6
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
EOF

# Deploy production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Cloud Deployment Options

#### AWS ECS/Fargate
```bash
# Install AWS CLI and ECS CLI
pip install awscli ecs-cli

# Configure ECS cluster
ecs-cli configure --cluster nurse-api --region us-west-2 --default-launch-type FARGATE

# Create ECS task definition from docker-compose
ecs-cli compose --project-name nurse-api service up --cluster nurse-api
```

#### Google Cloud Run
```dockerfile
# Dockerfile.cloudrun - optimized for Cloud Run
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use Cloud Run's PORT environment variable
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

```bash
# Deploy to Cloud Run
gcloud run deploy nurse-conversation-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

#### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nurse-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nurse-api
  template:
    metadata:
      labels:
        app: nurse-api
    spec:
      containers:
      - name: api
        image: nurse-conversation:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: nurse-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: nurse-secrets
              key: redis-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Container Optimization

#### Multi-stage Build
```dockerfile
# Dockerfile.optimized
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Copy dependencies from builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy application code
WORKDIR /app
COPY --chown=app:app . .

# Switch to non-root user
USER app
ENV PATH=/home/app/.local/bin:$PATH

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Health Checks
```yaml
# Enhanced health checks in docker-compose.yml
services:
  ambientai_be_api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  ambientai_be_postgres:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d nurse_conversations"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### Backup and Recovery

#### Database Backup
```bash
# Automated backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker-compose exec -T postgres pg_dump -U postgres nurse_conversations > "$BACKUP_DIR/backup_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/backup_$DATE.sql"

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: backup_$DATE.sql.gz"
EOF

chmod +x backup.sh

# Schedule with cron
# 0 2 * * * /path/to/backup.sh
```

#### Data Recovery
```bash
# Restore from backup
gunzip backup_20240115_020000.sql.gz

# Stop services
docker-compose down

# Start only database
docker-compose up -d postgres

# Wait for database to be ready
sleep 10

# Restore data
docker-compose exec -T postgres psql -U postgres -d nurse_conversations < backup_20240115_020000.sql

# Start all services
docker-compose up -d
```

### Monitoring and Alerting

#### Prometheus Monitoring
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

#### Log Aggregation
```yaml
# Add to docker-compose.yml for centralized logging
services:
  fluentd:
    image: fluent/fluentd:v1.14-debian-1
    volumes:
      - ./fluent.conf:/fluentd/etc/fluent.conf
      - ./logs:/var/log/app
    ports:
      - "24224:24224"

  # Add logging driver to all services
  ambientai_be_api:
    logging:
      driver: fluentd
      options:
        fluentd-address: localhost:24224
        tag: nurse-api
```

## 🔮 Future Enhancements

### Planned Features
- **Real-time Processing**: WebSocket support for live conversation updates
- **Advanced Analytics**: Conversation quality metrics and insights dashboard
- **Multi-language Support**: Internationalization for global deployment
- **Cloud Integration**: Native AWS S3, Google Cloud Storage support
- **Advanced AI Models**: Fine-tuned models for medical conversations
- **Compliance Features**: HIPAA, GDPR compliance tools and audit trails

### Performance Improvements
- **Caching Layer**: Redis caching for frequently accessed data
- **CDN Integration**: Static asset delivery optimization
- **Database Optimization**: Query optimization and indexing strategies
- **GPU Acceleration**: CUDA support for faster transcription processing

### Security Enhancements
- **OAuth2/OIDC**: Enterprise authentication integration
- **API Rate Limiting**: Advanced rate limiting and throttling
- **Encryption**: End-to-end encryption for sensitive data
- **Audit Logging**: Comprehensive audit trail and compliance reporting

---

**🚀 Developed by Soumyadeep**

*Empowering healthcare with intelligent conversation processing and AI-driven documentation automation.*

**Contact:** Reach out to Soumyadeep for enterprise deployments, custom integrations, or technical consultations.