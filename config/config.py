"""
Configuration settings for Watchdog AI.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DICTIONARY_PATH = ASSETS_DIR / "data_dictionary.json"
SAMPLE_DATA_DIR = ASSETS_DIR / "sample_data"

# App settings
APP_NAME = os.getenv("APP_NAME", "Watchdog AI")
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# AI settings
AI_CONFIG = {
    "model": os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229"),
    "max_tokens": int(os.getenv("MAX_TOKENS", "4096")),
    "temperature": 0.2
}

# Email settings
EMAIL_CONFIG = {
    "sendgrid_api_key": os.getenv("SENDGRID_API_KEY"),
    "notification_email": os.getenv("NOTIFICATION_EMAIL", "notifications@yourdomain.com"),
    "enable_notifications": os.getenv("ENABLE_NOTIFICATIONS", "false").lower() in ("true", "1", "t")
}

# Supabase settings
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
UPLOAD_BUCKET_NAME = os.getenv("UPLOAD_BUCKET_NAME", "watchdog-uploads")
PARSED_DATA_BUCKET_NAME = os.getenv("PARSED_DATA_BUCKET_NAME", "watchdog-parsed-data")

# Database settings
DB_CONFIG = {
    "enable_supabase": os.getenv("ENABLE_SUPABASE", "false").lower() in ("true", "1", "t"),
    "supabase_url": SUPABASE_URL,
    "supabase_key": SUPABASE_KEY,
    "upload_bucket": UPLOAD_BUCKET_NAME,
    "parsed_data_bucket": PARSED_DATA_BUCKET_NAME
}

# File processing settings
FILE_PROCESSING = {
    "max_file_size_mb": int(os.getenv("MAX_UPLOAD_SIZE_MB", 50)),
    "allowed_extensions": {
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".pdf": "application/pdf",
        ".txt": "text/plain"
    },
    "enable_pdf_processing": True
}

# UI Configuration
UI_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "🐕",
    "layout": "wide",
    "show_logo": True,
    "company_name": os.getenv("COMPANY_NAME", "Your Dealership")
}

def get_config():
    """Get the complete configuration dictionary."""
    return {
        "app_name": APP_NAME,
        "debug": DEBUG,
        "base_dir": str(BASE_DIR),
        "assets_dir": str(ASSETS_DIR),
        "data_dictionary_path": str(DATA_DICTIONARY_PATH),
        "ui": UI_CONFIG,
        "ai": AI_CONFIG,
        "email": EMAIL_CONFIG,
        "file_processing": FILE_PROCESSING
    }

def validate_config():
    """Validate that all required config variables are set."""
    # Create assets directory if it doesn't exist
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a basic data dictionary if it doesn't exist
    if not DATA_DICTIONARY_PATH.exists():
        import json
        basic_dictionary = {
            "version": "1.0",
            "fields": {
                "date_fields": ["created_date", "modified_date", "sale_date"],
                "lead_fields": ["lead_source", "lead_type"],
                "customer_fields": ["first_name", "last_name", "email", "phone"]
            }
        }
        DATA_DICTIONARY_PATH.write_text(json.dumps(basic_dictionary, indent=2))
    
    return True