"""
Celery configuration for the Nurse Conversation Processing API.
Handles distributed task queue setup and configuration.
"""
import os
from celery import Celery
from celery.signals import worker_ready, worker_shutdown
from kombu import Queue
from loguru import logger
from typing import Dict, Optional
import re
try:
    from redis import Redis
    import redis
except Exception:
    Redis = None  # type: ignore
    redis = None  # type: ignore

from config import get_settings


# Get settings
settings = get_settings()
celery_settings = settings.celery_settings

# Create Celery app
celery_app = Celery(
    "nurse_conversation_processor",
    broker=celery_settings.broker_url,
    backend=celery_settings.result_backend,
    include=[
        "tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer=celery_settings.task_serializer,
    result_serializer=celery_settings.result_serializer,
    accept_content=celery_settings.accept_content,
    
    # Timezone
    timezone=celery_settings.timezone,
    enable_utc=celery_settings.enable_utc,
    
    # Task settings
    task_track_started=celery_settings.task_track_started,
    task_time_limit=celery_settings.task_time_limit,
    task_soft_time_limit=celery_settings.task_time_limit - 60,  # 1 minute before hard limit
    
    # Worker settings
    worker_concurrency=celery_settings.worker_concurrency,
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    worker_disable_rate_limits=False,
    
    # Result settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Task routing and queues
    task_default_queue="default",
    task_routes={
        "tasks.transcribe_audio": {"queue": "transcription"},
        "tasks.extract_qa_answers": {"queue": "qa_extraction"},
        "tasks.process_conversation_complete": {"queue": "processing"},
        "tasks.calculate_conversation_score": {"queue": "scoring"},
    },
    
    # Define queues with different priorities
    task_queues=[
        Queue("default", routing_key="default"),
        Queue("transcription", routing_key="transcription"),
        Queue("qa_extraction", routing_key="qa_extraction"),
        Queue("processing", routing_key="processing"),
        Queue("scoring", routing_key="scoring"),
        Queue("high_priority", routing_key="high_priority"),
    ],
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Error handling
    task_reject_on_worker_lost=True,
    task_acks_late=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-failed-tasks": {
            "task": "tasks.cleanup_failed_tasks",
            "schedule": 3600.0,  # Run every hour
            "options": {"queue": "maintenance"}
        },
        "update-system-metrics": {
            "task": "tasks.update_system_metrics",
            "schedule": 300.0,  # Run every 5 minutes
            "options": {"queue": "maintenance"}
        },
    },
)

# Add maintenance queue
celery_app.conf.task_queues.append(
    Queue("maintenance", routing_key="maintenance")
)


# Signal handlers
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handler for when worker is ready."""
    logger.info(f"Celery worker {sender} is ready")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handler for when worker is shutting down."""
    logger.info(f"Celery worker {sender} is shutting down")


# Custom task base class
class CallbackTask(celery_app.Task):
    """Base task class with callbacks for progress tracking."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Task success callback."""
        logger.info(f"Task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Task failure callback."""
        logger.error(f"Task {task_id} failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Task retry callback."""
        logger.warning(f"Task {task_id} is being retried: {exc}")
    
    def update_progress(self, current: int, total: int, message: str = ""):
        """Update task progress."""
        progress = (current / total * 100) if total > 0 else 0
        self.update_state(
            state="PROGRESS",
            meta={
                "current": current,
                "total": total,
                "progress": progress,
                "message": message
            }
        )


# Set the custom task base class
celery_app.Task = CallbackTask


# Health check task
@celery_app.task(name="health_check")
def health_check():
    """Simple health check task."""
    return {"status": "healthy", "worker": os.getpid()}


# Utility functions
def get_task_status(task_id: str) -> dict:
    """Get the status of a task."""
    result = celery_app.AsyncResult(task_id)

    # Base response
    def _progress_resp(state: str, info: Optional[dict] = None) -> Dict:
        info = info or {}
        return {
            "state": state,
            "progress": float(info.get("progress", 0)),
            "current": info.get("current", 0),
            "total": info.get("total", 1),
            "message": info.get("message", "Task is pending..."),
        }

    state = result.state or "PENDING"

    # Celery returns PENDING for both unknown-id and not-started. Try to disambiguate.
    is_known_somewhere = False
    try:
        inspect = celery_app.control.inspect(timeout=2)
        active = inspect.active() or {}
        scheduled = inspect.scheduled() or {}
        reserved = inspect.reserved() or {}
        revoked = inspect.revoked() or {}

        def _contains(task_maps: Dict[str, list]) -> bool:
            for tasks in task_maps.values():
                for t in tasks or []:
                    task = t if isinstance(t, dict) else {}
                    if task.get("id") == task_id:
                        return True
                    # scheduled uses different shape
                    if task.get("request", {}).get("id") == task_id:
                        return True
            return False

        is_known_somewhere = _contains(active) or _contains(scheduled) or _contains(reserved) or _contains(revoked)
    except Exception:
        # Ignore inspection errors
        pass

    # Check backend meta existence
    meta = None
    try:
        meta = result.backend.get_task_meta(task_id)
    except Exception:
        meta = None

    # Decide on NOT_FOUND
    if state == "PENDING" and not is_known_somewhere:
        # meta may be {'status': 'PENDING'} even if not found; try to detect Redis key absence
        not_found = True
        try:
            if isinstance(result.backend, object) and getattr(result.backend, "client", None):
                # Redis backend: key like 'celery-task-meta-<id>'
                client = result.backend.client  # type: ignore[attr-defined]
                key = f"celery-task-meta-{task_id}"
                exists = client.exists(key)
                not_found = (exists == 0)
        except Exception:
            pass

        if not_found:
            return {"state": "NOT_FOUND", "progress": 0, "message": "Task ID not found"}

    # Map other states
    if state == "PROGRESS":
        info = result.info if isinstance(result.info, dict) else {}
        return _progress_resp(state, info)
    if state == "SUCCESS":
        return {"state": state, "progress": 100.0, "result": result.result}
    if state in {"FAILURE", "REVOKED"}:
        return {"state": state, "progress": 0.0, "error": str(result.info)}

    # Default pending
    info = result.info if isinstance(result.info, dict) else {}
    return _progress_resp("PENDING", info)


def cancel_task(task_id: str) -> bool:
    """Cancel a task."""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return True
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        return False


def _redis_client() -> Optional["Redis"]:
    try:
        if Redis is None:
            return None
        return redis.from_url(get_settings().redis_url)  # type: ignore[attr-defined]
    except Exception:
        return None


def get_queue_length(queue_name: str = "default") -> int:
    """Get the length of a specific queue (Redis broker)."""
    try:
        client = _redis_client()
        if client is None:
            return 0
        # Celery uses list named after queue for Redis broker; default queue is 'celery'
        redis_queue_key = "celery" if queue_name in {"default", "celery"} else queue_name
        return int(client.llen(redis_queue_key))
    except Exception as e:
        logger.error(f"Failed to get queue length for {queue_name}: {e}")
        return 0


def get_queue_lengths(queue_names: Optional[list] = None) -> Dict[str, int]:
    """Get lengths for all configured queues."""
    try:
        names = queue_names or [
            "default",
            "transcription",
            "qa_extraction",
            "processing",
            "scoring",
            "high_priority",
        ]
        return {name: get_queue_length(name) for name in names}
    except Exception as e:
        logger.error(f"Failed to get queue lengths: {e}")
        return {}


def get_worker_stats() -> dict:
    """Get statistics about Celery workers."""
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active = inspect.active()
        scheduled = inspect.scheduled()
        
        return {
            "workers": list(stats.keys()) if stats else [],
            "worker_count": len(stats) if stats else 0,
            "stats": stats,
            "active_tasks": active,
            "scheduled_tasks": scheduled,
        }
    except Exception as e:
        logger.error(f"Failed to get worker stats: {e}")
        return {
            "workers": [],
            "worker_count": 0,
            "stats": {},
            "active_tasks": {},
            "scheduled_tasks": {},
        }


def purge_queue(queue_name: str) -> int:
    """Purge all messages from a queue."""
    try:
        result = celery_app.control.purge()
        logger.info(f"Purged queue {queue_name}: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to purge queue {queue_name}: {e}")
        return 0


# Task retry configuration
def get_retry_kwargs(max_retries: int = 3, countdown: int = 60):
    """Get standard retry configuration for tasks."""
    return {
        "autoretry_for": (Exception,),
        "retry_kwargs": {"max_retries": max_retries, "countdown": countdown},
        "retry_backoff": True,
        "retry_backoff_max": 700,
        "retry_jitter": True,
    }


# Custom exceptions
class TaskTimeoutError(Exception):
    """Raised when a task times out."""
    pass


class TaskCancelledError(Exception):
    """Raised when a task is cancelled."""
    pass


class InsufficientResourcesError(Exception):
    """Raised when there are insufficient resources to process a task."""
    pass


# Configuration validation
def validate_celery_config():
    """Validate Celery configuration."""
    errors = []
    
    if not celery_settings.broker_url:
        errors.append("CELERY_BROKER_URL is required")
    
    if not celery_settings.result_backend:
        errors.append("CELERY_RESULT_BACKEND is required")
    
    if celery_settings.worker_concurrency < 1:
        errors.append("Worker concurrency must be at least 1")
    
    if errors:
        raise ValueError(f"Celery configuration errors: {', '.join(errors)}")
    
    logger.info("Celery configuration validated successfully")


# Initialize configuration validation
try:
    validate_celery_config()
except ValueError as e:
    logger.error(f"Celery configuration validation failed: {e}")
    raise