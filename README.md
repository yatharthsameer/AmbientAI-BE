# AmbientAI-BE

> A comprehensive, scalable backend system for processing nurse-patient conversations with AI-powered transcription, intelligent Q&A extraction, and automated quality scoring.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/postgres-%23316192.svg?logo=postgresql&logoColor=white)](https://postgresql.org/)


## 🌟 What This System Does

AmbientAI-BE transforms raw audio recordings or text transcripts of nurse-patient conversations into structured, actionable medical information. It's designed for healthcare institutions, EMR systems, and medical documentation workflows.

### Key Capabilities

- **🎙️ Audio Transcription**: Converts audio files to text using OpenAI Whisper with timestamp precision
- **🧠 Intelligent Q&A Extraction**: Uses advanced AI (Gemini + DistilBERT + verification) to answer predefined medical questions
- **📊 Quality Scoring**: Automatically assesses conversation completeness and information quality
- **⚡ Asynchronous Processing**: Handles multiple files simultaneously with real-time progress tracking
- **🔗 RESTful API**: Clean, documented endpoints for easy integration
- **📈 Scalable Architecture**: Built with Docker, Celery, and modern async Python

## 🏥 Real-World Use Cases

- **EMR Integration**: Automatically populate patient records from recorded conversations
- **Quality Assurance**: Assess nursing documentation completeness and accuracy
- **Training & Education**: Analyze conversation patterns for nursing education programs
- **Compliance**: Ensure all required information is captured during patient interactions
- **Research**: Extract structured data from large volumes of healthcare conversations

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** or **Docker** (recommended)
- **4GB+ RAM** for local development
- **PostgreSQL** and **Redis** (or use Docker Compose)
- **API Keys** (optional but recommended):
  - Google Gemini API key for best Q&A results
  - OpenAI API key for fallback processing

### Option 1: Docker Setup (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd AmbientAI-BE

# Copy environment configuration
cp env.example .env
# Edit .env with your API keys (optional but recommended)

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View API documentation
open http://localhost:8000/docs
```

### Option 2: Local Development Setup

```bash
# Clone and setup
git clone <your-repo-url>
cd AmbientAI-BE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp env.example .env
# Edit .env with your database and API settings

# Start PostgreSQL and Redis (ensure they're running)
# Initialize database
python -c "from database import init_database; import asyncio; asyncio.run(init_database())"

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start Celery worker
celery -A celery_app worker --loglevel=info

# In another terminal, start Celery beat (optional, for periodic tasks)
celery -A celery_app beat --loglevel=info
```

## 📝 How to Use

### 1. Upload an Audio File

```bash
curl -X POST "http://localhost:8000/api/v1/uploads" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@nurse_conversation.wav" \
  -F "process_immediately=true"
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "original_filename": "nurse_conversation.wav",
  "status": "pending",
  "transcription_task_id": "celery-task-id-123",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 2. Process Text Directly (No Audio)

```bash
curl -X POST "http://localhost:8000/api/v1/text-processing" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning, Mrs. Johnson. How are you feeling today? I have been having chest pain since yesterday. On a scale of 1 to 10, how would you rate your pain? I would say it is about a 7. Are you taking any medications? Yes, I take Metformin for diabetes and Lisinopril for blood pressure.",
    "custom_questions": [
      {
        "id": "custom_1",
        "question": "What time of day did symptoms start?",
        "category": "timeline"
      }
    ]
  }'
```

### 3. Check Processing Status

```bash
curl "http://localhost:8000/api/v1/tasks/{task_id}"
```

**Response:**
```json
{
  "task_id": "celery-task-id-123",
  "status": "running",
  "progress": 65.0,
  "current_step": "Extracting Q&A answers...",
  "message": "Processing medical information extraction"
}
```

### 4. Get Complete Results

```bash
curl "http://localhost:8000/api/v1/uploads/{upload_id}"
```

**Response:**
```json
{
  "upload": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "original_filename": "nurse_conversation.wav",
    "status": "completed",
    "duration_seconds": 180.5,
    "processing_completed_at": "2024-01-15T10:35:30Z"
  },
  "transcription": {
    "full_text": "Good morning, Mrs. Johnson. How are you feeling today? I have been having chest pain since yesterday...",
    "language": "en",
    "confidence_score": 0.94,
    "segments": [
      {
        "start": 0.0,
        "end": 3.2,
        "text": "Good morning, Mrs. Johnson."
      },
      {
        "start": 3.5,
        "end": 7.8,
        "text": "How are you feeling today?"
      }
    ]
  },
  "qa_results": [
    {
      "question_id": "chief_complaint",
      "question_text": "What is the patient's chief complaint?",
      "answer_text": "chest pain",
      "confidence_score": 0.95,
      "timestamp_start": 15.2,
      "timestamp_end": 17.8,
      "is_confident": true,
      "is_manual_review_required": false
    },
    {
      "question_id": "pain_level",
      "question_text": "What is the patient's pain level on a scale of 1 to 10?",
      "answer_text": "7 out of 10",
      "confidence_score": 0.92,
      "timestamp_start": 45.1,
      "timestamp_end": 48.3,
      "is_confident": true
    },
    {
      "question_id": "medications",
      "question_text": "What medications is the patient currently taking?",
      "answer_text": "Metformin for diabetes and Lisinopril for blood pressure",
      "confidence_score": 0.88,
      "timestamp_start": 65.4,
      "timestamp_end": 72.1,
      "is_confident": true
    }
  ],
  "score": {
    "completeness_score": 0.85,
    "confidence_score": 0.92,
    "information_density_score": 0.78,
    "patient_info_score": 0.90,
    "medical_history_score": 0.88,
    "assessment_score": 0.82,
    "treatment_score": 0.75,
    "questions_answered": 10,
    "questions_total": 12,
    "high_confidence_answers": 8,
    "answers_requiring_review": 2
  }
}
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```bash
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/nurse_conversations

# Redis Configuration (for Celery)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# AI Service API Keys (optional but recommended)
GEMINI_API_KEY=your_google_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Whisper Configuration
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
WHISPER_DEVICE=cpu  # Options: cpu, cuda

# Application Settings
SECRET_KEY=your-super-secret-key-here
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
WORKERS=4

# File Upload Settings
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=104857600  # 100MB in bytes
ALLOWED_AUDIO_FORMATS=mp3,wav,m4a,ogg,flac

# Q&A Model Settings
QA_MODEL=distilbert-base-cased-distilled-squad
MAX_QUESTION_LENGTH=512
MAX_CONTEXT_LENGTH=4096
```

### Supported Audio Formats

- **MP3** (.mp3)
- **WAV** (.wav) 
- **M4A** (.m4a)
- **OGG** (.ogg)
- **FLAC** (.flac)

### Predefined Medical Questions

The system comes with 12 predefined questions covering common nursing documentation needs:

1. **Patient Information**: Name, age
2. **Medical History**: Chief complaint, symptoms
3. **Medications**: Current medications, allergies
4. **Assessment**: Vital signs, pain level
5. **Diagnosis**: Medical diagnosis
6. **Treatment**: Treatment plan
7. **Discharge**: Discharge instructions, follow-up plans

You can also add custom questions via the API.

## 🏗️ Architecture Overview 🔗 [here](ARCHITECTURE.md)

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   FastAPI       │    │   Celery        │
│   (Frontend)    │◄──►│   (main.py)     │◄──►│   Workers       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   PostgreSQL    │    │     Redis       │
                       │   (Database)    │    │   (Broker)      │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   File Storage  │    │   AI Services   │
                       │   (/uploads)    │    │ Gemini/OpenAI   │
                       └─────────────────┘    └─────────────────┘
```

### Processing Pipeline

```
Audio Upload → Transcription → Q&A Extraction → Scoring → Results
     │              │              │             │          │
     ▼              ▼              ▼             ▼          ▼
File Storage → Whisper Model → AI Services → Algorithm → Database
```

### Key Features

- **Asynchronous Processing**: Non-blocking task execution with progress tracking
- **Queue-based Architecture**: Separate queues for different task types
- **AI Service Integration**: Multiple AI providers with fallback mechanisms  
- **Comprehensive Logging**: Structured logging with correlation IDs
- **Health Monitoring**: Built-in health checks and monitoring endpoints
- **Scalable Design**: Horizontal scaling support for high-volume processing

For detailed architecture diagrams and technical specifications, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 📊 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/uploads` | Upload audio file for processing |
| `POST` | `/api/v1/text-processing` | Process text directly (no audio) |
| `GET` | `/api/v1/uploads/{id}` | Get complete processing results |
| `GET` | `/api/v1/tasks/{task_id}` | Check task processing status |
| `GET` | `/api/v1/uploads` | List all uploads with pagination |
| `POST` | `/api/v1/tasks/{task_id}/cancel` | Cancel a running task |

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/api/v1/questions` | Get predefined questions |
| `GET` | `/api/v1/admin/stats` | System statistics (admin) |
| `GET` | `/docs` | Interactive API documentation |
| `GET` | `/redoc` | Alternative API documentation |

### Test Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/test/gemini` | Test Gemini API integration |
| `POST` | `/api/v1/test/medical-extraction` | Test medical info extraction |

## 🧠 AI Services & Models

### Primary AI Stack

1. **Transcription**: OpenAI Whisper
   - Models: `tiny`, `base`, `small`, `medium`, `large`
   - Supports 99+ languages
   - Timestamp-accurate transcription
   - CPU and GPU support

2. **Q&A Extraction**: Hybrid approach
   - **Primary**: Google Gemini 2.0 Flash
   - **Secondary**: DistilBERT (local processing)
   - **Verification**: Custom verification service
   - **Fallback**: OpenAI GPT-4

3. **Quality Scoring**: Custom algorithms
   - Completeness assessment
   - Confidence scoring
   - Information density analysis
   - Category-specific scoring

### AI Service Configuration

```python
# Gemini Configuration
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.1
GEMINI_MAX_TOKENS=4096

# OpenAI Configuration  
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=4096

# Local Model Configuration
QA_MODEL=distilbert-base-cased-distilled-squad
WHISPER_MODEL=base
```

## 🔍 Monitoring & Observability

### Built-in Monitoring

- **Health Checks**: `/health` endpoint with service status
- **Task Monitoring**: Flower UI at `http://localhost:5555`
- **Structured Logging**: JSON logs with correlation IDs
- **Performance Metrics**: Response times, queue lengths, error rates

### Key Metrics to Monitor

- **API Performance**: Response times, error rates, throughput
- **Task Processing**: Queue lengths, processing times, success rates  
- **Database**: Connection pool usage, query performance
- **AI Services**: API response times, quota usage, error rates
- **System Resources**: CPU, memory, disk usage

### Log Locations

- **Application Logs**: `./logs/app.log`
- **Container Logs**: `docker-compose logs -f [service]`
- **Database Logs**: PostgreSQL container logs
- **Task Logs**: Celery worker logs

## 🛠️ Development

### Local Development Setup

```bash
# Clone repository
git clone <repo-url>
cd AmbientAI-BE

# Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup development database
docker-compose -f docker-compose.dev.yml up -d postgres redis

# Run database migrations
python -c "from database import init_database; import asyncio; asyncio.run(init_database())"

# Start development server with hot reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker (in another terminal)
celery -A celery_app worker --loglevel=debug --concurrency=1
```

### Development Tools

- **Hot Reloading**: FastAPI with `--reload` flag
- **Database Admin**: Adminer at `http://localhost:8080`
- **Redis Management**: Redis Commander at `http://localhost:8081`  
- **Jupyter Notebooks**: Available at `http://localhost:8888`
- **Task Monitoring**: Flower at `http://localhost:5555`

### Testing

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_api.py

# Test specific functionality
python -m pytest -k "test_upload"
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 .
mypy .

# Security scan
bandit -r .
```

## 🚀 Deployment

### Production Deployment

```bash
# Clone repository
git clone <repo-url>
cd AmbientAI-BE

# Configure production environment
cp env.example .env
# Edit .env with production settings

# Deploy with Docker Compose
docker-compose -f docker-compose.yml up -d

# Verify deployment
docker-compose ps
curl http://localhost:8000/health
```

For comprehensive Docker deployment instructions, see [README-Docker.md](README-Docker.md).

### Scaling for Production

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  api:
    deploy:
      replicas: 4
  
  worker:
    deploy:
      replicas: 6
      
  transcription-worker:
    deploy:
      replicas: 3
```

### Security Considerations

- **Environment Variables**: Never commit `.env` files
- **API Keys**: Use secure key management systems
- **Database**: Use strong passwords and connection encryption
- **File Uploads**: Validate file types and scan for malware
- **Rate Limiting**: Implement API rate limiting for production
- **HTTPS**: Use SSL/TLS for all external communication

### Performance Optimization

- **Database**: Add indexes for frequently queried fields
- **Caching**: Implement Redis caching for frequent queries
- **CDN**: Use CDN for static file serving
- **Load Balancing**: Use nginx or cloud load balancers
- **GPU Support**: Enable GPU acceleration for Whisper transcription

## 🔧 Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check Docker status
docker-compose ps

# View service logs
docker-compose logs api
docker-compose logs worker

# Restart services
docker-compose restart
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL health
docker-compose exec postgres pg_isready -U postgres

# Reset database (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

#### 3. Celery Tasks Not Processing
```bash
# Check worker status
docker-compose exec api celery -A celery_app inspect active

# Check Redis connection  
docker-compose exec redis redis-cli ping

# Restart workers
docker-compose restart worker
```

#### 4. AI Service Errors
```bash
# Check API keys in .env file
# Verify API quotas and rate limits
# Check service logs for specific error messages
docker-compose logs api | grep -i "gemini\|openai"
```

#### 5. File Upload Issues
```bash
# Check upload directory permissions
ls -la uploads/

# Verify file size limits
# Check allowed file formats in configuration
```

### Debug Mode

Enable debug logging:
```bash
# In .env file
DEBUG=true
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart
```

### Performance Issues

- Monitor resource usage: `docker stats`
- Check queue lengths in Flower UI
- Review database query performance
- Consider scaling workers or API instances

## 📈 Roadmap

### Short Term (Next 1-2 months)
- [ ] WebSocket support for real-time updates
- [ ] Advanced conversation analytics
- [ ] Custom model fine-tuning capabilities
- [ ] Enhanced security features

### Medium Term (2-4 months)
- [ ] Multi-language conversation support
- [ ] Cloud storage integration (S3, GCS)
- [ ] Advanced reporting and dashboards
- [ ] Mobile app integration

### Long Term (6+ months)
- [ ] HIPAA compliance features
- [ ] Machine learning model improvements
- [ ] Enterprise SSO integration
- [ ] Advanced workflow automation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Soumyadeep** - *Lead Developer & System Architect*

- 🏗️ **System Architecture**: Designed scalable microservices architecture with Docker & Celery
- 🤖 **AI Integration**: Implemented hybrid AI pipeline with Gemini, OpenAI, and DistilBERT
- 📊 **Database Design**: Created comprehensive PostgreSQL schema for medical data
- 🐳 **DevOps**: Docker containerization, multi-environment deployment strategies
- 📚 **Documentation**: Comprehensive technical documentation and deployment guides

## 🙏 Acknowledgments

- **OpenAI** for Whisper transcription model
- **Google** for Gemini AI services  
- **Hugging Face** for DistilBERT and transformers
- **FastAPI** team for the excellent web framework
- **Celery** project for distributed task processing

## 📞 Support

Need help? Here's how to get support:

1. **Check the Documentation**: Review this README and the [Architecture Guide](ARCHITECTURE.md)
2. **Search Issues**: Look through existing [GitHub Issues](https://github.com/your-repo/issues)
3. **Check Logs**: Review application and service logs for error details
4. **Health Check**: Verify system status at `/health` endpoint
5. **Contact Soumyadeep**: Reach out for technical support and consultation
6. **Create Issue**: Open a new GitHub issue with detailed information

### Support Information Template

When reporting issues, please include:

```
**Environment:**
- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12]
- Python Version: [e.g., 3.11.2]  
- Docker Version: [e.g., 20.10.17]
- Deployment Method: [Docker Compose, Local Development, etc.]

**Issue Description:**
[Clear description of the problem]

**Steps to Reproduce:**
1. [First step]
2. [Second step]  
3. [Third step]

**Expected Behavior:**
[What you expected to happen]

**Actual Behavior:**  
[What actually happened]

**Logs:**
[Relevant log entries]

**Additional Context:**
[Any other relevant information]
```

---

**🚀 Happy Processing!** 🎉

Built with ⚡ for healthcare professionals and developers working to improve patient care through better documentation and analysis.

*Empowering healthcare with intelligent conversation processing and AI-driven documentation automation.*

---


