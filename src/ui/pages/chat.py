"""
Chat interface for Watchdog AI.

This module provides the Copilot chat interface to interact with Claude.
"""
import streamlit as st
import pandas as pd
import os
import time
import json
from typing import Dict, List, Any, Optional
from anthropic import Anthropic

# Import configuration
from config.config import (
    AI_CONFIG, DATA_DICTIONARY_PATH, APP_NAME, EMAIL_CONFIG
)

# Import AI components
from src.ai.claude_client import ClaudeClient
from src.ai.prompts import get_chat_prompt, get_data_validation_prompt

def render_chat_page():
    """Render the chat page."""
    st.title("Copilot Chat")
    st.write("Chat with your AI assistant about your dealership data.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = "I'm here to help you analyze your dealership data. What would you like to know?"
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

def _get_data_context() -> Dict[str, Any]:
    """
    Get context about the available data for Claude.
    
    Returns:
        dict: Data context for Claude
    """
    context = {}
    
    # If we have sample data, add it to the context
    if "sample_data" in st.session_state and "sample_results" in st.session_state:
        df = st.session_state.sample_data
        results = st.session_state.sample_results
        
        # Format column mapping for context
        column_mapping = {}
        proc_results = results.get("processing_results", {})
        if "column_mapping" in proc_results:
            column_mapping = proc_results["column_mapping"]
        
        # Extract some basic metrics
        metrics = {}
        try:
            # Add row count
            metrics["row_count"] = len(df)
            
            # Add date range if available
            date_cols = [col for col in df.columns if "date" in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    date_range = {
                        "min": df[date_col].min().strftime('%Y-%m-%d') if not pd.isna(df[date_col].min()) else "N/A",
                        "max": df[date_col].max().strftime('%Y-%m-%d') if not pd.isna(df[date_col].max()) else "N/A"
                    }
                    metrics["date_range"] = date_range
                except:
                    pass
            
            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if "price" in col.lower() or "cost" in col.lower():
                    metrics[f"{col}_avg"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else 0
                    metrics[f"{col}_min"] = float(df[col].min()) if not pd.isna(df[col].min()) else 0
                    metrics[f"{col}_max"] = float(df[col].max()) if not pd.isna(df[col].max()) else 0
            
            # Add validation results if available
            validation = results.get("validation_results", {})
            if validation:
                metrics["is_valid"] = validation.get("is_valid", True)
                metrics["issues"] = validation.get("issues", [])
                metrics["warnings"] = validation.get("warnings", [])
        
        except Exception as e:
            metrics["error"] = str(e)
        
        # Combine all context for this dataset
        dataset_name = results.get("filename", "uploaded_data")
        context[dataset_name] = {
            "columns": list(df.columns),
            "mapped_columns": column_mapping,
            "row_count": len(df),
            "detected_template": proc_results.get("detected_template", "Unknown"),
            "template_confidence": proc_results.get("template_confidence", 0),
            "metrics": metrics,
            "sample_data": df.head(5).to_dict(orient="records")
        }
    
    return context

def _load_data_dictionary() -> Dict[str, Any]:
    """
    Load the data dictionary for terminology and field descriptions.
    
    Returns:
        dict: Data dictionary
    """
    # Start with a basic dictionary that will be used if the file doesn't exist
    data_dict = {
        "terms": {
            "lead": "A potential customer who has shown interest",
            "VIN": "Vehicle Identification Number, a unique code for each vehicle",
            "stock_number": "Dealer's internal inventory identifier for a vehicle",
            "days_in_stock": "Number of days a vehicle has been in inventory"
        },
        "metrics": {
            "average_days_in_stock": "Industry average is 60 days",
            "lead_conversion_rate": "Industry average is 8-10%",
            "cost_per_lead": "Industry average is $25-40 per lead"
        }
    }
    
    # Try to load from the dictionary file
    try:
        if os.path.exists(DATA_DICTIONARY_PATH):
            with open(DATA_DICTIONARY_PATH, 'r') as f:
                file_dict = json.load(f)
                
                # We'll just use the fields for now
                if "field_mappings" in file_dict:
                    field_terms = {}
                    for category, fields in file_dict["field_mappings"].items():
                        category_name = category.replace("_fields", "")
                        for field in fields:
                            field_terms[field] = f"A {category_name} field"
                    
                    # Add to our terms
                    data_dict["terms"].update(field_terms)
    except Exception as e:
        # If there's an error, just use the default dictionary
        pass
    
    return data_dict

if __name__ == "__main__":
    # This allows running just this page for development
    render_chat_page()