"""
Upload page for Watchdog AI.

This page handles file uploads, validation, and processing.
"""
import streamlit as st
import pandas as pd
import os
import time
import json
from pathlib import Path
import tempfile

# Import configuration
from config.config import (
    FILE_PROCESSING, DATA_DICTIONARY_PATH, SAMPLE_DATA_DIR, APP_NAME
)

# Import data processing components
from src.data.parsers.csv_parser import CSVParser
from src.data.template_mapper import TemplateMapper
from src.data.validators.data_validator import DataValidator

# Import services (will implement properly in next step)
# from src.services.storage import upload_to_supabase

def is_valid_file(file):
    """Check if the file is valid for upload."""
    # Check file size
    max_size_bytes = FILE_PROCESSING["max_file_size_mb"] * 1024 * 1024
    if file.size > max_size_bytes:
        return False, f"File is too large. Maximum size is {FILE_PROCESSING['max_file_size_mb']} MB."
    
    # Check file extension
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext not in FILE_PROCESSING["allowed_extensions"]:
        allowed_exts = ", ".join(FILE_PROCESSING["allowed_extensions"].keys())
        return False, f"File type '{file_ext}' is not supported. Supported types: {allowed_exts}"
    
    return True, ""

def process_file(file):
    """
    Process an uploaded file using the appropriate parser.
    """
    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
        temp_file.write(file.getbuffer())
        temp_path = temp_file.name
    
    try:
        # Initialize the right parser based on file type
        file_ext = os.path.splitext(file.name)[1].lower()
        
        # Process CSV and Excel files
        if file_ext in ['.csv', '.txt', '.xls', '.xlsx']:
            parser = CSVParser()
            df, results = parser.parse_file(temp_path)
            
            # Validate the data
            validator = DataValidator()
            validation_results = validator.validate(df)
            
            # Combine results
            combined_results = {
                "processing_results": results,
                "validation_results": validation_results,
                "filename": file.name,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records")
            }
            
            return True, df, combined_results
            
        # PDF handling will be available if enabled
        elif file_ext == '.pdf':
            if FILE_PROCESSING.get("enable_pdf_processing", False):
                # PDF parser will be implemented in future
                from src.data.parsers.pdf_parser import PDFParser
                parser = PDFParser()
                tables, results = parser.parse_file(temp_path)
                
                if not tables:
                    return False, None, "No tables found in the PDF file."
                
                # Use the first table found
                df = tables[0]
                return True, df, {
                    "processing_results": results,
                    "filename": file.name,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "preview": df.head(5).to_dict(orient="records"),
                    "tables_found": len(tables)
                }
            else:
                return False, None, "PDF processing is not enabled."
        else:
            return False, None, f"File type '{file_ext}' is not supported."
    
    except Exception as e:
        return False, None, f"Error processing file: {str(e)}"
    
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def render_upload_page():
    """Render the upload page."""
    st.title("Upload Files")
    st.write("Upload your dealership data files for analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file containing your dealership data."
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display basic information
            st.write("File successfully uploaded!")
            st.write(f"Number of rows: {len(df)}")
            st.write(f"Number of columns: {len(df.columns)}")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    # This allows running just this page for development
    render_upload_page()