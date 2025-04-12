"""
Utility modules for the Watchdog AI application.
"""

from .exceptions import (
    # Base exception
    WatchdogError,
    
    # Validation exceptions
    ValidationError,
    MissingRequiredFieldError,
    FormatValidationError,
    DataTypeError,
    OutlierError,
    DuplicateDataError,
    MissingValueError,
    
    # AI service exceptions
    AIServiceError,
    APIError,
    PromptError,
    
    # Storage exceptions
    StorageError,
    FileAccessError,
    DataSerializationError
)