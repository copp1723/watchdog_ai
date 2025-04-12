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

# App settings
APP_NAME = os.getenv("APP_NAME", "Watchdog AI")
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

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
        "ui": UI_CONFIG
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