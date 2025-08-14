"""
Helper utility functions for the Nurse Conversation Processing API.
"""
import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional, Union
from fastapi import UploadFile
from pydub import AudioSegment
from loguru import logger


async def save_upload_file(upload_file: UploadFile, upload_dir: Union[str, Path]) -> Path:
    """
    Save uploaded file to disk with a unique filename.
    
    Args:
        upload_file: FastAPI UploadFile object
        upload_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_extension = Path(upload_file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = upload_dir / unique_filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as buffer:
        content = await upload_file.read()
        await buffer.write(content)
    
    logger.info(f"File saved: {file_path} ({len(content)} bytes)")
    return file_path


def get_file_duration(file_path: Union[str, Path]) -> Optional[float]:
    """
    Get audio file duration in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds or None if unable to determine
    """
    try:
        audio = AudioSegment.from_file(str(file_path))
        duration = len(audio) / 1000.0  # Convert milliseconds to seconds
        return duration
    except Exception as e:
        logger.warning(f"Could not get duration for {file_path}: {e}")
        return None


def create_directory(path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of the created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.warning(f"Could not get size for {file_path}: {e}")
        return 0


def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """
    Validate if file has an allowed extension.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (without dots)
        
    Returns:
        True if extension is allowed, False otherwise
    """
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower().lstrip('.')
    return file_extension in [ext.lower() for ext in allowed_extensions]


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = f"file_{uuid.uuid4().hex[:8]}"
    
    return sanitized


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def calculate_processing_eta(file_size: int, duration: Optional[float] = None) -> Optional[float]:
    """
    Estimate processing time based on file characteristics.
    
    Args:
        file_size: File size in bytes
        duration: Audio duration in seconds
        
    Returns:
        Estimated processing time in seconds
    """
    # These are rough estimates - adjust based on your system performance
    base_time = 30  # Base processing time in seconds
    
    # Time based on file size (assuming ~10MB per minute processing)
    size_factor = file_size / (10 * 1024 * 1024)  # MB
    
    # Time based on duration (assuming real-time transcription + QA)
    duration_factor = duration * 1.5 if duration else 0
    
    estimated_time = base_time + (size_factor * 60) + duration_factor
    return max(estimated_time, 30)  # Minimum 30 seconds


def cleanup_old_files(directory: Union[str, Path], max_age_hours: int = 24) -> int:
    """
    Clean up old files from a directory.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep (in hours)
        
    Returns:
        Number of files deleted
    """
    import time
    
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    current_time = time.time()
    cutoff_time = current_time - (max_age_hours * 3600)
    
    deleted_count = 0
    for file_path in directory.iterdir():
        if file_path.is_file():
            try:
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not delete file {file_path}: {e}")
    
    logger.info(f"Cleaned up {deleted_count} old files from {directory}")
    return deleted_count


def ensure_upload_directory(upload_dir: Union[str, Path]) -> Path:
    """
    Ensure upload directory exists and is writable.
    
    Args:
        upload_dir: Path to upload directory
        
    Returns:
        Path object of the upload directory
        
    Raises:
        PermissionError: If directory is not writable
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Test if directory is writable
    test_file = upload_dir / f"test_write_{uuid.uuid4().hex[:8]}.tmp"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Upload directory {upload_dir} is not writable: {e}")
    
    return upload_dir


def get_mime_type(file_path: Union[str, Path]) -> str:
    """
    Get MIME type for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string
    """
    import mimetypes
    
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"


def is_audio_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is an audio file based on MIME type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is audio, False otherwise
    """
    mime_type = get_mime_type(file_path)
    return mime_type.startswith("audio/")


def generate_unique_id() -> str:
    """
    Generate a unique identifier.
    
    Returns:
        Unique ID string
    """
    return str(uuid.uuid4())


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename for filesystem storage.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Safe filename
    """
    # Sanitize the filename
    safe_name = sanitize_filename(filename)
    
    # Truncate if too long
    if len(safe_name) > max_length:
        name_part, ext_part = os.path.splitext(safe_name)
        available_length = max_length - len(ext_part)
        safe_name = name_part[:available_length] + ext_part
    
    return safe_name