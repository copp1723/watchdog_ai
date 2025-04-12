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
    
    # Add custom CSS for better chat styling
    st.markdown("""
    <style>
        /* User message styling */
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
            border-radius: 10px;
        }
        
        /* Style for user messages */
        [data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"] {
            background-color: #2E7BF6 !important;
            color: white !important;
        }
        
        /* Style for assistant messages */
        [data-testid="stChatMessage"][data-testid*="assistant"] [data-testid="stChatMessageContent"] {
            background-color: #F2F2F7 !important;
            border: 1px solid #E5E5EA;
        }
        
        /* Add timestamp styling */
        .message-timestamp {
            font-size: 0.7rem;
            color: #8E8E93;
            text-align: right;
            margin-top: -5px;
            padding-right: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a welcome message
        current_time = time.strftime("%H:%M")
        welcome_msg = "👋 Hello! I'm your Watchdog AI assistant. I can help you analyze your dealership data, explain metrics, and answer questions. How can I help you today?"
        st.session_state.messages.append({
            "role": "assistant", 
            "content": welcome_msg,
            "timestamp": current_time
        })
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp", "")
        
        with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🐕"):
            st.markdown(content)
            if timestamp:
                st.markdown(f"<div class='message-timestamp'>{timestamp}</div>", unsafe_allow_html=True)
    
    # Add suggested questions
    if len(st.session_state.messages) <= 2:  # Only show suggestions at the beginning
        st.markdown("##### Suggested Questions:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 What are my top lead sources?"):
                handle_user_input("What are my top lead sources?")
            if st.button("🚗 Show me aging inventory analysis"):
                handle_user_input("Show me aging inventory analysis")
        
        with col2:
            if st.button("📈 How has my lead-to-sale ratio changed?"):
                handle_user_input("How has my lead-to-sale ratio changed?")
            if st.button("🔍 Who is my top performing sales rep?"):
                handle_user_input("Who is my top performing sales rep?")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        handle_user_input(prompt)

def handle_user_input(prompt: str):
    """Handle user input and generate response."""
    current_time = time.strftime("%H:%M")
    
    # Add user message to chat history (with timestamp)
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": current_time
    })
    
    # Display user message in the UI (Streamlit auto-displays new messages)
    
    # Get data context for AI response
    context = _get_data_context()
    data_dict = _load_data_dictionary()
    
    # Initialize Claude client
    try:
        claude = ClaudeClient()
        
        # Display assistant response with a typing indicator
        with st.chat_message("assistant", avatar="🐕"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Thinking...")
            
            # Get response from Claude (in production, this would use the actual AI)
            
            # For the MVP, use a simple response based on the question
            time.sleep(1.5)  # Simulate AI thinking
            
            # Simple response logic (would be replaced with actual Claude call)
            if "lead source" in prompt.lower():
                response = "Based on your data, your top lead sources are:\n\n1. Website (45 leads, 37.5%)\n2. Phone (32 leads, 26.7%)\n3. Walk-in (18 leads, 15%)\n\nYour website is performing particularly well compared to industry benchmarks. Consider investing more in your digital marketing channels."
            elif "inventory" in prompt.lower() or "aging" in prompt.lower():
                response = "Your inventory aging analysis shows:\n\n- Average days in stock: 42 days (industry benchmark: 35 days)\n- Aged inventory (>60 days): 21% of total inventory\n\nThis is concerning as your aged inventory percentage has increased from 18% to 21% in the past period. Consider implementing a pricing strategy for vehicles approaching the 60-day threshold."
            elif "ratio" in prompt.lower() or "conversion" in prompt.lower():
                response = "Your lead-to-sale ratio has improved significantly:\n\n- Current: 15% (up from 12%)\n- This represents a 25% improvement\n- Industry benchmark: 12%\n\nYour team is performing above the industry benchmark. I recommend documenting your current lead handling process to ensure consistent application."
            elif "sales rep" in prompt.lower() or "performance" in prompt.lower():
                response = "Your top performing sales reps are:\n\n1. John Smith (35 leads)\n2. Jane Doe (32 leads)\n3. Bob Johnson (28 leads)\n4. Alice Brown (25 leads)\n\nJohn Smith is your highest performer, handling 29% more leads than your lowest performer."
            else:
                response = "I'd be happy to help with that. To provide the most accurate information, could you upload your dealership data or specify which metrics you're interested in? I can analyze lead performance, inventory health, website traffic, and more."
    except Exception as e:
        response = f"I apologize, but I encountered an error while processing your request. Please check your API configuration or try again later."
    
    # Update the message placeholder with the actual response
    message_placeholder.markdown(response)
    
    # Add the response to chat history with timestamp
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "timestamp": time.strftime("%H:%M")
    })

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