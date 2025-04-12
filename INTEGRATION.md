# Watchdog AI Improvements Integration Guide

This document provides instructions for integrating the improved components into the main Watchdog AI application.

## 1. Custom Exceptions Integration

1. Copy the custom exceptions module to the main project:
   ```bash
   cp -r Desktop/watchdog_ai_improvements/src/utils/ watchdog_ai/src/
   ```

2. The improved DataValidator has already been integrated into the main project.

## 2. Enhanced Template Mapper Integration

1. Copy the enhanced template mapper to the project:
   ```bash
   cp Desktop/watchdog_ai_improvements/src/data/enhanced_template_mapper.py watchdog_ai/src/data/
   ```

2. Update imports in relevant files to use the enhanced mapper:
   ```python
   # Replace
   from src.data.template_mapper import TemplateMapper
   # With
   from src.data.enhanced_template_mapper import EnhancedTemplateMapper
   ```

3. When initializing the mapper, use the enhanced version:
   ```python
   # Replace
   mapper = TemplateMapper()
   # With
   from src.ai.claude_client import ClaudeClient
   claude_client = ClaudeClient()
   mapper = EnhancedTemplateMapper(claude_client=claude_client)
   ```

## 3. Interactive Validation Dashboard Integration

1. Copy the validation dashboard components:
   ```bash
   cp Desktop/watchdog_ai_improvements/src/ui/pages/dashboard/validation_dashboard.py watchdog_ai/src/ui/pages/dashboard/
   cp Desktop/watchdog_ai_improvements/src/ui/pages/validation.py watchdog_ai/src/ui/pages/
   ```

2. Update the dashboard __init__.py file to include the validation dashboard:
   ```bash
   cp Desktop/watchdog_ai_improvements/src/ui/pages/dashboard/__init__.py watchdog_ai/src/ui/pages/dashboard/
   ```

3. Update the main app.py file to include the validation page:
   ```python
   # Add to imports
   from src.ui.pages.validation import render as render_validation
   
   # Add to pages dictionary
   pages = {
       "Upload": render_upload,
       "Chat": render_chat,
       "Dashboard": render_dashboard,
       "Validation": render_validation  # Add this line
   }
   ```

## 4. Email Digest System Integration

1. Create the configuration module:
   ```bash
   mkdir -p watchdog_ai/src/config
   cp Desktop/watchdog_ai_improvements/src/config/*.py watchdog_ai/src/config/
   cp Desktop/watchdog_ai_improvements/config/default.json watchdog_ai/config/
   cp Desktop/watchdog_ai_improvements/.env.example watchdog_ai/
   ```

2. Copy the digest service:
   ```bash
   cp Desktop/watchdog_ai_improvements/src/services/digest.py watchdog_ai/src/services/
   ```

3. Copy the digest UI component:
   ```bash
   cp Desktop/watchdog_ai_improvements/src/ui/pages/dashboard/digest.py watchdog_ai/src/ui/pages/dashboard/
   ```

4. Update the dashboard __init__.py file to include the digest page:
   ```python
   # Add to imports in watchdog_ai/src/ui/pages/dashboard/__init__.py
   from .digest import render_digest_page
   
   # Add to __all__
   __all__ = [
       'render_metrics_dashboard',
       'render_insights_dashboard',
       'render_validation_dashboard',
       'render_digest_page'  # Add this line
   ]
   ```

5. Update the main app.py file to include the digest page:
   ```python
   # Add to imports
   from .pages.dashboard.digest import render_digest_page
   
   # Add to pages dictionary
   pages = {
       "Upload": render_upload,
       "Chat": render_chat,
       "Dashboard": render_dashboard,
       "Validation": render_validation,
       "Email Digest": render_digest_page  # Add this line
   }
   ```

6. Configure SendGrid API:
   - Create an account at SendGrid if you don't have one
   - Generate an API key with email sending permissions
   - Add the API key to your .env file:
     ```
     SENDGRID_API_KEY=your_sendgrid_api_key_here
     SENDER_EMAIL=your_sender_email@example.com
     ```

## 5. Test Suite Integration

1. Copy the test files to the main project:
   ```bash
   mkdir -p watchdog_ai/tests/data/validators
   mkdir -p watchdog_ai/tests/services
   cp -r Desktop/watchdog_ai_improvements/tests/data/* watchdog_ai/tests/data/
   cp -r Desktop/watchdog_ai_improvements/tests/services/* watchdog_ai/tests/services/
   ```

2. Run the tests:
   ```bash
   cd watchdog_ai
   python -m unittest discover
   ```

## 6. Configuration Updates

1. If using an existing Claude API key, update the API client configuration to share the key between applications.

## Usage Examples

### Enhanced Error Handling

```python
from src.data.validators.data_validator import DataValidator
from src.utils.exceptions import ValidationError, MissingRequiredFieldError

try:
    validator = DataValidator(raise_on_error=True)
    results = validator.validate_critical(df)
except MissingRequiredFieldError as e:
    print(f"Missing fields: {e.context['missing_fields']}")
    print(f"Suggestions: {e.context['suggested_fields']}")
except ValidationError as e:
    print(f"Validation failed: {e.message}")
```

### AI-Powered Field Mapping

```python
from src.data.enhanced_template_mapper import EnhancedTemplateMapper
from src.ai.claude_client import ClaudeClient

# Initialize with Claude client
claude_client = ClaudeClient()
mapper = EnhancedTemplateMapper(claude_client=claude_client)

# Get suggestions for missing categories
suggestions = mapper.suggest_field_mappings(
    df, 
    ["date_fields", "lead_source_fields"]
)

print(f"Suggested date fields: {suggestions['date_fields']}")
print(f"Suggested lead source fields: {suggestions['lead_source_fields']}")
```

### Interactive Validation Dashboard

```python
import streamlit as st
from src.data.validators.data_validator import DataValidator
from src.ui.pages.dashboard.validation_dashboard import render_validation_dashboard

# Validate data
validator = DataValidator()
validation_results = validator.validate(df)

# Render dashboard
render_validation_dashboard(validation_results, df)
```

### Email Digest System

```python
from src.services.digest import DigestGenerator
from datetime import datetime, timedelta

# Initialize generator
generator = DigestGenerator()

# Generate digest content
start_date = datetime.now() - timedelta(days=7)
end_date = datetime.now()
digest_html = generator.generate_digest_content(
    df=dataframe,
    dealer_name="Sample Dealership",
    date_range=(start_date, end_date)
)

# Send digest email
result = generator.send_digest_email(
    recipient_email="manager@dealership.com",
    digest_html=digest_html,
    dealer_name="Sample Dealership",
    attach_data=dataframe  # Optional
)

if result["success"]:
    print("Digest sent successfully!")
else:
    print(f"Failed to send digest: {result.get('error')}")
```