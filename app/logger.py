import json
import logging
from logging.handlers import RotatingFileHandler
import os
import uuid
from datetime import datetime

class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "event": getattr(record, "event", "general"),
            "message": record.getMessage(),
            "user_id": getattr(record, "user_id", None),
            "session_id": getattr(record, "session_id", None),
            "meta": getattr(record, "meta", {})
        }
        return json.dumps(log_record, cls=UUIDEncoder)

_logger = None

def get_logger():
    global _logger
    if _logger is not None:
        return _logger
    
    logger = logging.getLogger("varsapradaya")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if not logger.handlers:
        formatter = JsonFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        os.makedirs("logs", exist_ok=True)
        file_handler = RotatingFileHandler(
            "logs/app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    _logger = logger
    return _logger
