## System Architecture Overview

This document describes the end-to-end architecture for the Nurse Conversation Processing API, including core components, processing pipelines, and the data model. Diagrams are written in Mermaid so they render directly in most code hosts.

### Key Components
- **FastAPI (`main.py`)**: HTTP API for uploads, status, and results
- **Celery (`celery_app.py`, `tasks.py`)**: Background processing with queue routing and progress updates
- **Workers**: Dedicated queues for transcription, QA extraction, processing, and scoring
- **Services**: Transcription (Whisper), QA Extraction (Hybrid: Gemini + DistilBERT + Final Verification)
- **Data**: PostgreSQL for persistence, Redis as Celery broker/result backend, local `uploads/` and `logs/`

### Component Diagram
```mermaid
graph TD
  %% Nodes
  subgraph "Client"
    user["User / Integrator"]
  end

  subgraph "Edge / Proxy"
    nginx["Nginx (optional) :80 / :443"]
  end

  subgraph "Application Layer"
    api["FastAPI App :8000\nmain.py"]
    celery_api["Celery API & Config\ncelery_app.py"]
  end

  subgraph "Workers & Schedulers"
    worker_default["Celery Worker\nqueues: default, processing, qa_extraction, scoring"]
    worker_tx["Transcription Worker\nqueues: transcription, high_priority"]
    beat["Celery Beat\nperiodic: cleanup_failed_tasks, update_system_metrics"]
    flower["Flower UI :5555\nmonitoring"]
  end

  subgraph "Data & Storage"
    redis["Redis :6379\nBroker + Result Backend"]
    pg["PostgreSQL :5432\nconversation_* tables"]
    uploads["Uploads dir /app/uploads"]
    logs["Logs /app/logs"]
  end

  subgraph "Tasks"
    t_pipeline["process_conversation_complete\nqueue: processing"]
    t_tx["transcribe_audio\nqueue: transcription"]
    t_qa["extract_qa_answers\nqueue: qa_extraction"]
    t_score["calculate_conversation_score\nqueue: scoring"]
    t_step["process_conversation_step"]
    t_cb["workflow_callback"]
    t_cleanup["cleanup_failed_tasks (beat)"]
    t_metrics["update_system_metrics (beat)"]
  end

  subgraph "Services"
    svc_tx["TranscriptionService\nWhisper"]
    svc_qa["QAExtractionService\nHybrid: Gemini + DistilBERT + Verification"]
    svc_gemini["GeminiService"]
    svc_openai["OpenAIService"]
    svc_distil["DistilBERTService"]
    svc_verify["FinalVerificationService"]
  end

  %% Client to API
  user -->|"Upload audio / text"| nginx
  nginx --> api
  user -.->|"Direct (no proxy)"| api

  %% API interactions
  api -->|"Save file"| uploads
  api <--> |"CRUD"| pg
  api -->|"Enqueue Celery task (delay)"| t_pipeline
  api --> celery_api

  %% Task routing through broker
  t_pipeline -. "routes to 'processing'" .-> redis
  t_tx -. "routes to 'transcription'" .-> redis
  t_qa -. "routes to 'qa_extraction'" .-> redis
  t_score -. "routes to 'scoring'" .-> redis
  t_cleanup -. "routes to 'maintenance' (beat)" .-> redis
  t_metrics -. "routes to 'maintenance' (beat)" .-> redis

  %% Workers pull from broker
  worker_default <--> redis
  worker_tx <--> redis
  beat <--> redis
  flower <--> redis

  %% Workers execute tasks and hit services/DB
  worker_default --> t_pipeline
  worker_tx --> t_tx
  worker_default --> t_qa
  worker_default --> t_score
  worker_default --> t_step
  worker_default --> t_cb

  %% Task to services
  t_tx --> svc_tx
  t_qa --> svc_qa
  svc_qa --> svc_gemini
  svc_qa --> svc_openai
  svc_qa --> svc_distil
  svc_qa --> svc_verify

  %% Persist results
  t_tx -->|"write transcription"| pg
  t_qa -->|"write Q&A results"| pg
  t_score -->|"write score"| pg
  t_pipeline -->|"return task ids"| api

  %% Workflow progression
  t_tx --> t_cb --> t_step --> t_qa --> t_cb --> t_step --> t_score --> t_cb

  %% Observability
  api --> logs
  worker_default --> logs
  worker_tx --> logs
  beat --> logs
  flower --> logs

  %% User monitoring
  user -.->|"Flower UI :5555"| flower

  %% Task status polling
  user -->|"GET /api/v1/tasks/{task_id}"| api
  api <--> redis

  %% Results fetch
  user -->|"GET /api/v1/uploads/{upload_id}"| api
  api --> pg
```

Notes:
- The API writes uploads to `uploads/` and persists metadata/results to PostgreSQL.
- Celery routes different tasks to queues for isolation and scaling.
- Redis is both the broker and result backend for task state/progress.

## Audio Upload Pipeline (Sequence)

This sequence shows the asynchronous pipeline for audio files: upload → transcription → Q&A extraction → scoring.

```mermaid
sequenceDiagram
  autonumber
  participant U as Client
  participant A as FastAPI (main.py)
  participant F as File Storage (/app/uploads)
  participant R as Redis (Broker/Backend)
  participant Wt as Celery Worker (transcription)
  participant Wd as Celery Worker (default)
  participant DB as PostgreSQL
  participant T as TranscriptionService (Whisper)
  participant Q as QAExtractionService (Hybrid)
  participant G as GeminiService
  participant O as OpenAIService
  participant D as DistilBERTService
  participant V as FinalVerificationService

  U->>A: POST /api/v1/uploads (audio)
  A->>F: Save file
  A->>DB: Insert ConversationUpload (status=pending)
  A->>R: Enqueue process_conversation_complete
  A-->>U: 202 Accepted (+task id)

  Note over R,Wd: Queue "processing"
  R-->>Wd: process_conversation_complete
  Wd->>R: transcribe_audio.delay(upload_id)
  Note over R,Wt: Queue "transcription"
  R-->>Wt: transcribe_audio
  Wt->>DB: Update Upload(status=processing)
  Wt->>T: transcribe_audio(file_path)
  T-->>Wt: {text, segments, language, ...}
  Wt->>DB: Insert ConversationTranscription
  Wt->>DB: Update Upload(status=completed)
  Wt->>R: workflow_callback("transcription")

  Note over R,Wd: Queue "default"
  R-->>Wd: workflow_callback
  Wd->>R: process_conversation_step("qa_extraction")

  Note over R,Wd: Queue "qa_extraction"
  R-->>Wd: extract_qa_answers
  Wd->>DB: Load transcription
  Wd->>Q: extract_multiple_answers(questions, context, segments)
  Q->>G: extract_medical_info (if available)
  Q->>D: extract_medical_info
  Q->>V: verify_and_clean_results
  alt fallback to OpenAI
    Q->>O: extract_medical_info
  end
  Q-->>Wd: list of answers (+confidence/timestamps)
  Wd->>DB: Insert QuestionAnswer[*]
  Wd->>R: workflow_callback("qa_extraction")

  Note over R,Wd: Queue "default"
  R-->>Wd: workflow_callback
  Wd->>R: process_conversation_step("scoring")

  Note over R,Wd: Queue "scoring"
  R-->>Wd: calculate_conversation_score
  Wd->>DB: Read QuestionAnswer[*]
  Wd->>DB: Insert ConversationScore
  Wd->>R: workflow_callback("scoring")

  U->>A: GET /api/v1/tasks/{task_id}
  A->>R: inspect AsyncResult
  R-->>A: {state, progress, meta}
  A-->>U: TaskProgressResponse

  U->>A: GET /api/v1/uploads/{upload_id}
  A->>DB: Load Upload + Transcription + QA
  DB-->>A: Aggregated data
  A-->>U: CompleteConversationResponse
```

## Text-only Pipeline (Sequence)

When text is provided (no audio), the system creates a synthetic transcription and proceeds with Q&A and scoring.

```mermaid
sequenceDiagram
  autonumber
  participant U as Client
  participant A as FastAPI (main.py)
  participant R as Redis (Broker/Backend)
  participant Wd as Celery Worker (default)
  participant T as TranscriptionService (text_only)
  participant Q as QAExtractionService (Hybrid)
  participant DB as PostgreSQL

  U->>A: POST /api/v1/text-processing { text }
  A->>DB: Insert ConversationUpload(status=pending)
  A->>R: process_text_only.delay(upload_id, text, custom_questions)
  A-->>U: 202 Accepted (+task id)

  Note over R,Wd: Queue "default"
  R-->>Wd: process_text_only
  Wd->>T: transcribe_text_only(text)
  T-->>Wd: {text, segments, ...}
  Wd->>DB: Insert ConversationTranscription
  Wd->>R: extract_qa_answers.delay(upload_id, text, segments)
  Wd->>R: calculate_conversation_score.delay(upload_id)

  Note over R,Wd: Queue "qa_extraction" / "scoring"
  R-->>Wd: extract_qa_answers
  Wd->>Q: extract_multiple_answers(questions, context=text, segments)
  Q-->>Wd: answers[]
  Wd->>DB: Insert QuestionAnswer[*]

  R-->>Wd: calculate_conversation_score
  Wd->>DB: Read QuestionAnswer[*]
  Wd->>DB: Insert ConversationScore

  U->>A: GET /api/v1/tasks/{task_id}
  A->>R: inspect AsyncResult
  R-->>A: {state, progress, meta}
  A-->>U: TaskProgressResponse

  U->>A: GET /api/v1/uploads/{upload_id}
  A->>DB: Load Upload + Transcription + QA + Score
  DB-->>A: Aggregated data
  A-->>U: CompleteConversationResponse
```

## Database ER Model

The ER diagram aligns with the SQLAlchemy models in `database_models.py` and the Pydantic schemas in `schemas.py`.

```mermaid
erDiagram
  CONVERSATION_UPLOADS ||--o{ CONVERSATION_TRANSCRIPTIONS : has
  CONVERSATION_UPLOADS ||--o{ QUESTION_ANSWERS : has
  CONVERSATION_UPLOADS ||--o| CONVERSATION_SCORES : has_one
  CONVERSATION_UPLOADS ||--o{ PROCESSING_JOBS : has

  CONVERSATION_UPLOADS {
    UUID id PK
    string original_filename
    string file_path
    int file_size
    string content_type
    float duration_seconds
    string status
    string transcription_task_id
    string qa_extraction_task_id
    string error_message
    datetime processing_started_at
    datetime processing_completed_at
    datetime created_at
    datetime updated_at
  }

  CONVERSATION_TRANSCRIPTIONS {
    UUID id PK
    UUID upload_id FK
    text full_text
    json segments
    string language
    string model_used
    float processing_time_seconds
    float confidence_score
    datetime created_at
    datetime updated_at
  }

  QUESTION_ANSWERS {
    UUID id PK
    UUID upload_id FK
    string question_id
    text question_text
    string category
    text answer_text
    float confidence_score
    int context_start_char
    int context_end_char
    float timestamp_start
    float timestamp_end
    text context_snippet
    string model_used
    bool is_confident
    bool is_manual_review_required
    datetime created_at
    datetime updated_at
  }

  PROCESSING_JOBS {
    UUID id PK
    string job_type
    string task_id
    UUID upload_id FK
    string status
    float progress_percentage
    string current_step
    int total_steps
    datetime started_at
    datetime completed_at
    json result_data
    text error_message
    text error_traceback
    string worker_name
    datetime created_at
    datetime updated_at
  }

  CONVERSATION_SCORES {
    UUID id PK
    UUID upload_id FK
    float completeness_score
    float confidence_score
    float information_density_score
    float patient_info_score
    float medical_history_score
    float assessment_score
    float treatment_score
    int questions_answered
    int questions_total
    int high_confidence_answers
    int answers_requiring_review
    float transcription_quality_score
    datetime scores_calculated_at
    datetime created_at
    datetime updated_at
  }
```

## Deployment & Ports
- **FastAPI**: `:8000` (see `docker-compose.yml` → service `ambientai_be_api`)
- **Flower**: `:5555` for task monitoring (`ambientai_be_flower`)
- **Redis**: `:6379` broker/result backend (`ambientai_be_redis`)
- **PostgreSQL**: `:5432` (`ambientai_be_postgres`)
- **Nginx (optional)**: `:80`, `:443` (`ambientai_be_nginx`)
- Dev extras: Adminer `:8080`, Redis Commander `:8081`, Jupyter `:8888` (see `docker-compose.dev.yml`)

## Task Routing (Queues)
- `tasks.transcribe_audio` → `transcription`
- `tasks.extract_qa_answers` → `qa_extraction`
- `tasks.process_conversation_complete` → `processing`
- `tasks.calculate_conversation_score` → `scoring`
- Periodic: `cleanup_failed_tasks`, `update_system_metrics` → `maintenance`

## How to read and update
- The diagrams reflect the code in `main.py`, `celery_app.py`, `tasks.py`, `database_models.py`, and `services/*`.
- Edit flows by updating tasks in `tasks.py` and their routes in `celery_app.py`.
- Add new entities in `database_models.py` and extend the ER diagram as needed.

---

**📐 Architecture designed by Soumyadeep**

*System architect and lead developer 


