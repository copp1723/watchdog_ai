"""
Custom exception classes for the Watchdog AI application.

This module defines a hierarchy of custom exceptions for different components of the
Watchdog AI system, enabling more structured error handling and recovery mechanisms.
"""
from typing import Any, Dict, List, Optional


class WatchdogError(Exception):
    """Base exception class for all Watchdog AI errors"""
    
    def __init__(self, 
                message: str, 
                error_code: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
                recovery_hint: Optional[str] = None):
        """
        Initialize a WatchdogError exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic identification
            context: Optional dict with contextual information about the error
            recovery_hint: Optional hint for how to recover from this error
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.recovery_hint = recovery_hint
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to a dictionary representation.
        
        Returns:
            Dict representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "recovery_hint": self.recovery_hint
        }


# Data Validation Exceptions
class ValidationError(WatchdogError):
    """Base exception for all data validation errors"""
    
    def __init__(self, 
                message: str, 
                error_code: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
                recovery_hint: Optional[str] = None,
                **kwargs):
        """
        Initialize a ValidationError exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic identification
            context: Optional dict with contextual information about the error
            recovery_hint: Optional hint for how to recover from this error
            **kwargs: Additional context fields to include
        """
        context = context or {}
        context.update(kwargs)
        super().__init__(message, error_code, context, recovery_hint)


class MissingRequiredFieldError(ValidationError):
    """Exception raised when required fields are missing"""
    
    def __init__(self, 
                message: str, 
                missing_fields: List[str] = None,
                suggested_fields: Dict[str, List[str]] = None,
                **kwargs):
        """
        Initialize a MissingRequiredFieldError exception.
        
        Args:
            message: Human-readable error message
            missing_fields: List of field names that are missing
            suggested_fields: Optional dict mapping missing categories to potential field matches
            **kwargs: Additional context fields to include
        """
        context = {
            "missing_fields": missing_fields or [],
            "suggested_fields": suggested_fields or {}
        }
        context.update(kwargs)
        recovery_hint = "Ensure all required fields are present in the data."
        if suggested_fields:
            recovery_hint += " Consider using the suggested field mappings."
        super().__init__(message, "MISSING_REQUIRED_FIELD", context, recovery_hint, **kwargs)


class FormatValidationError(ValidationError):
    """Exception raised when field format validation fails"""
    
    def __init__(self, 
                message: str, 
                field_name: str,
                invalid_count: int = 0,
                invalid_percentage: float = 0.0,
                format_type: Optional[str] = None,
                samples: List[str] = None,
                **kwargs):
        """
        Initialize a FormatValidationError exception.
        
        Args:
            message: Human-readable error message
            field_name: The name of the field with format issues
            invalid_count: Count of invalid values
            invalid_percentage: Percentage of invalid values
            format_type: Type of format (email, phone, VIN, etc.)
            samples: Sample invalid values
            **kwargs: Additional context fields to include
        """
        context = {
            "field_name": field_name,
            "invalid_count": invalid_count,
            "invalid_percentage": invalid_percentage,
            "format_type": format_type,
            "samples": samples or []
        }
        context.update(kwargs)
        recovery_hint = f"Check the format of values in the '{field_name}' field."
        super().__init__(message, "FORMAT_VALIDATION_ERROR", context, recovery_hint, **kwargs)


class DataTypeError(ValidationError):
    """Exception raised when data type inconsistencies are found"""
    
    def __init__(self, 
                message: str, 
                field_name: str,
                expected_type: str,
                found_types: Dict[str, int] = None,
                samples: List[Any] = None,
                **kwargs):
        """
        Initialize a DataTypeError exception.
        
        Args:
            message: Human-readable error message
            field_name: The name of the field with type issues
            expected_type: Expected data type
            found_types: Dict mapping type names to counts
            samples: Sample values with incorrect types
            **kwargs: Additional context fields to include
        """
        context = {
            "field_name": field_name,
            "expected_type": expected_type,
            "found_types": found_types or {},
            "samples": samples or []
        }
        context.update(kwargs)
        recovery_hint = f"Ensure '{field_name}' field contains only {expected_type} values."
        super().__init__(message, "DATA_TYPE_ERROR", context, recovery_hint, **kwargs)


class OutlierError(ValidationError):
    """Exception raised when significant outliers are detected"""
    
    def __init__(self, 
                message: str, 
                field_name: str,
                outlier_count: int = 0,
                outlier_percentage: float = 0.0,
                min_value: Optional[float] = None,
                max_value: Optional[float] = None,
                outlier_samples: List[Any] = None,
                **kwargs):
        """
        Initialize an OutlierError exception.
        
        Args:
            message: Human-readable error message
            field_name: The name of the field with outliers
            outlier_count: Count of outlier values
            outlier_percentage: Percentage of outlier values
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            outlier_samples: Sample outlier values
            **kwargs: Additional context fields to include
        """
        context = {
            "field_name": field_name,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "min_value": min_value,
            "max_value": max_value,
            "outlier_samples": outlier_samples or []
        }
        context.update(kwargs)
        
        recovery_hint = f"Review outlier values in '{field_name}' field."
        if min_value is not None and max_value is not None:
            recovery_hint += f" Expected range: {min_value} to {max_value}."
            
        super().__init__(message, "OUTLIER_ERROR", context, recovery_hint, **kwargs)


class DuplicateDataError(ValidationError):
    """Exception raised when excessive duplicate data is detected"""
    
    def __init__(self, 
                message: str, 
                duplicate_count: int = 0,
                duplicate_percentage: float = 0.0,
                duplicate_indices: List[int] = None,
                **kwargs):
        """
        Initialize a DuplicateDataError exception.
        
        Args:
            message: Human-readable error message
            duplicate_count: Count of duplicate rows
            duplicate_percentage: Percentage of duplicate rows
            duplicate_indices: Indices of duplicate rows
            **kwargs: Additional context fields to include
        """
        context = {
            "duplicate_count": duplicate_count,
            "duplicate_percentage": duplicate_percentage,
            "duplicate_indices": duplicate_indices or []
        }
        context.update(kwargs)
        recovery_hint = "Consider removing or merging duplicate entries."
        super().__init__(message, "DUPLICATE_DATA_ERROR", context, recovery_hint, **kwargs)


class MissingValueError(ValidationError):
    """Exception raised when a field has excessive missing values"""
    
    def __init__(self, 
                message: str, 
                field_name: str,
                missing_count: int = 0,
                missing_percentage: float = 0.0,
                **kwargs):
        """
        Initialize a MissingValueError exception.
        
        Args:
            message: Human-readable error message
            field_name: The name of the field with missing values
            missing_count: Count of missing values
            missing_percentage: Percentage of missing values
            **kwargs: Additional context fields to include
        """
        context = {
            "field_name": field_name,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage
        }
        context.update(kwargs)
        recovery_hint = f"Consider imputing missing values in '{field_name}' or verifying data collection."
        super().__init__(message, "MISSING_VALUE_ERROR", context, recovery_hint, **kwargs)


# AI Service Exceptions
class AIServiceError(WatchdogError):
    """Base exception for AI service errors"""
    pass


class APIError(AIServiceError):
    """Exception raised when there is an error calling the AI API"""
    
    def __init__(self, 
                message: str, 
                status_code: Optional[int] = None,
                api_response: Optional[Dict[str, Any]] = None,
                **kwargs):
        """
        Initialize an APIError exception.
        
        Args:
            message: Human-readable error message
            status_code: HTTP status code if applicable
            api_response: Raw API response if available
            **kwargs: Additional context fields to include
        """
        context = {
            "status_code": status_code,
            "api_response": api_response or {}
        }
        context.update(kwargs)
        recovery_hint = "Check API credentials and try again later."
        super().__init__(message, "AI_API_ERROR", context, recovery_hint)


class PromptError(AIServiceError):
    """Exception raised when there's an issue with prompt construction or execution"""
    
    def __init__(self, 
                message: str, 
                prompt_name: Optional[str] = None,
                prompt_params: Optional[Dict[str, Any]] = None,
                **kwargs):
        """
        Initialize a PromptError exception.
        
        Args:
            message: Human-readable error message
            prompt_name: Name of the prompt template that failed
            prompt_params: Parameters passed to the prompt
            **kwargs: Additional context fields to include
        """
        context = {
            "prompt_name": prompt_name,
            "prompt_params": prompt_params or {}
        }
        context.update(kwargs)
        recovery_hint = "Review prompt parameters and ensure they match the expected format."
        super().__init__(message, "PROMPT_ERROR", context, recovery_hint)


# Storage Service Exceptions
class StorageError(WatchdogError):
    """Base exception for storage service errors"""
    pass


class FileAccessError(StorageError):
    """Exception raised when file access operations fail"""
    
    def __init__(self, 
                message: str, 
                file_path: str,
                operation: str = "access",
                **kwargs):
        """
        Initialize a FileAccessError exception.
        
        Args:
            message: Human-readable error message
            file_path: Path to the file that couldn't be accessed
            operation: The operation that failed (read, write, etc.)
            **kwargs: Additional context fields to include
        """
        context = {
            "file_path": file_path,
            "operation": operation
        }
        context.update(kwargs)
        recovery_hint = f"Check file permissions and existence of '{file_path}'."
        super().__init__(message, "FILE_ACCESS_ERROR", context, recovery_hint)


class DataSerializationError(StorageError):
    """Exception raised when data serialization or deserialization fails"""
    
    def __init__(self, 
                message: str, 
                data_format: str = "unknown",
                operation: str = "serialize",
                **kwargs):
        """
        Initialize a DataSerializationError exception.
        
        Args:
            message: Human-readable error message
            data_format: Format of the data (json, csv, etc.)
            operation: The operation that failed (serialize, deserialize)
            **kwargs: Additional context fields to include
        """
        context = {
            "data_format": data_format,
            "operation": operation
        }
        context.update(kwargs)
        recovery_hint = f"Check data format and structure for {data_format} compatibility."
        super().__init__(message, "DATA_SERIALIZATION_ERROR", context, recovery_hint)