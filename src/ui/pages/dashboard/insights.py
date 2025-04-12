"""
Dashboard insights visualization for Watchdog AI.

This module provides the AI-generated insights visualization for the dashboard.
"""
import streamlit as st
import pandas as pd
import re
from typing import Dict, List, Any, Optional

# Import Claude client
from src.ai.claude_client import ClaudeClient


def clean_markdown(text: str) -> str:
    """
    Clean and standardize markdown formatting
    
    Args:
        text: Raw markdown text from Claude
        
    Returns:
        Cleaned markdown text
    """
    # Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Ensure headers have space after #
    text = re.sub(r'(#+)([^ #])', r'\1 \2', text)
    
    # Clean up bullet points
    text = re.sub(r'\s*\*\s*', '* ', text)
    
    return text


def render_insights_section(digest_data: Dict[str, Any], claude_client: Optional[ClaudeClient] = None):
    """
    Render the AI insights section
    
    Args:
        digest_data: Digest data with metrics
        claude_client: Optional Claude client
    """
    st.title("AI Insights")
    
    # Check if we already have insights text
    insights_text = digest_data.get("text", "")
    
    if not insights_text and claude_client:
        # Generate insights on demand
        with st.spinner("Generating insights..."):
            try:
                # Import here to avoid circular imports
                from src.ai.prompts import get_digest_prompt
                
                # Get the prompts
                prompts = get_digest_prompt(digest_data)
                
                # Generate insights
                insights_text = claude_client.answer_question(
                    prompts["user"], digest_data
                )
                
                # Update the digest data
                digest_data["text"] = insights_text
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                insights_text = ""
    
    if insights_text:
        # Clean and display insights
        cleaned_text = clean_markdown(insights_text)
        st.markdown(cleaned_text)
    else:
        st.info("No AI insights available. Please check your Claude API configuration.")
    
    # Add option to regenerate
    if claude_client and st.button("Regenerate Insights"):
        with st.spinner("Regenerating insights..."):
            try:
                # Import here to avoid circular imports
                from src.ai.prompts import get_digest_prompt
                
                # Get the prompts
                prompts = get_digest_prompt(digest_data)
                
                # Generate insights
                insights_text = claude_client.answer_question(
                    prompts["user"], digest_data
                )
                
                # Update the digest data
                digest_data["text"] = insights_text
                
                # Clean and display insights
                cleaned_text = clean_markdown(insights_text)
                st.markdown(cleaned_text)
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")


def render_anomalies_section(anomalies: List[Dict[str, Any]]):
    """
    Render anomalies detected in data
    
    Args:
        anomalies: List of anomalies from digest data
    """
    if not anomalies:
        st.info("No significant changes detected in your data.")
        return
    
    st.subheader("Significant Changes Detected")
    
    # Sort anomalies by severity
    high_anomalies = [a for a in anomalies if a.get("severity") == "high"]
    medium_anomalies = [a for a in anomalies if a.get("severity") == "medium"]
    
    # Display high severity anomalies
    if high_anomalies:
        st.markdown("##### Critical Changes")
        
        for anomaly in high_anomalies:
            # Format the anomaly for display
            metric = anomaly.get("metric", "").replace("_", " ").title()
            direction = anomaly.get("direction", "")
            pct_change = anomaly.get("percent_change", 0)
            current = anomaly.get("current_value", 0)
            previous = anomaly.get("previous_value", 0)
            
            # Format the percentage change
            pct_display = f"{abs(pct_change) * 100:.1f}%" if -1 < pct_change < 1 else f"{abs(pct_change):.1f}x"
            
            # Determine color
            color = "red" if (
                "decrease" in direction and "ratio" not in metric.lower() and "day" not in metric.lower()
            ) or (
                "increase" in direction and ("day" in metric.lower() or "aged" in metric.lower())
            ) else "green"
            
            # Create the anomaly card
            with st.container(border=True):
                st.markdown(f"### {metric}")
                st.markdown(f"<span style='color:{color};font-size:1.2em;font-weight:bold;'>{direction.title()} by {pct_display}</span>", unsafe_allow_html=True)
                
                # Compare current vs previous
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Current", current)
                with cols[1]:
                    st.metric("Previous", previous)
    
    # Display medium severity anomalies
    if medium_anomalies:
        st.markdown("##### Notable Changes")
        
        # Create columns for medium anomalies
        cols = st.columns(2)
        
        for i, anomaly in enumerate(medium_anomalies):
            col = cols[i % 2]
            
            with col:
                # Format the anomaly for display
                metric = anomaly.get("metric", "").replace("_", " ").title()
                direction = anomaly.get("direction", "")
                pct_change = anomaly.get("percent_change", 0)
                current = anomaly.get("current_value", 0)
                previous = anomaly.get("previous_value", 0)
                
                # Format the percentage change
                pct_display = f"{abs(pct_change) * 100:.1f}%" if -1 < pct_change < 1 else f"{abs(pct_change):.1f}x"
                
                # Create the anomaly card
                with st.container(border=True):
                    st.markdown(f"#### {metric}")
                    st.markdown(f"{direction.title()} by {pct_display}")
                    st.markdown(f"**Current:** {current}")
                    st.markdown(f"**Previous:** {previous}")


def render_recommendations_section(digest_data: Dict[str, Any], claude_client: Optional[ClaudeClient] = None):
    """
    Render AI recommendations section
    
    Args:
        digest_data: Digest data with metrics
        claude_client: Optional Claude client
    """
    st.subheader("AI Recommendations")
    
    # First check if we have pre-generated recommendations
    recommendations = digest_data.get("recommendations", [])
    
    if not recommendations and claude_client:
        # Generate recommendations on demand
        with st.spinner("Generating recommendations..."):
            try:
                prompt = """Based on the metrics and anomalies in this data, provide 3-5 specific, actionable recommendations for the dealership. Focus on addressing any concerning trends and capitalizing on positive changes. Format each recommendation as a concise bullet point with a clear action item."""
                
                response = claude_client.answer_question(prompt, digest_data)
                
                # Extract bullet points
                bullet_pattern = r'[\*\-]\s*(.*?)(?=[\*\-]|$)'
                matches = re.findall(bullet_pattern, response, re.DOTALL)
                
                if matches:
                    recommendations = [match.strip() for match in matches if match.strip()]
                else:
                    recommendations = [response]
                
                # Update the digest data
                digest_data["recommendations"] = recommendations
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                recommendations = []
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations):
            with st.container(border=True):
                st.markdown(f"##### Recommendation {i+1}")
                st.markdown(rec)
    else:
        st.info("No recommendations available.")
        
        # Show a button to generate recommendations
        if claude_client and st.button("Generate Recommendations"):
            # This will trigger the generation logic on the next render
            st.experimental_rerun()


def render_insights_dashboard(digest_data: Dict[str, Any]):
    """
    Render the complete insights dashboard
    
    Args:
        digest_data: Digest data with metrics, anomalies, and insights
    """
    # Initialize Claude client for on-demand generation
    try:
        claude_client = ClaudeClient()
    except:
        claude_client = None
        st.warning("Claude API not configured. Some features will be disabled.")
    
    # Create tabs for different insight views
    tab1, tab2, tab3 = st.tabs([
        "Weekly Digest", "Anomalies", "Recommendations"
    ])
    
    with tab1:
        render_insights_section(digest_data, claude_client)
    
    with tab2:
        render_anomalies_section(digest_data.get("anomalies", []))
    
    with tab3:
        render_recommendations_section(digest_data, claude_client)


if __name__ == "__main__":
    # Demo data for testing
    demo_data = {
        "metrics": {
            "total_leads": 120,
            "lead_to_appointment": 0.32,
            "lead_to_sale": 0.15,
            "appointment_to_sale": 0.45,
            "lead_sources": {
                "Website": 45,
                "Phone": 32,
                "Walk-in": 18,
                "Referral": 15,
                "Social Media": 10
            },
            "rep_performance": {
                "John Smith": 35,
                "Jane Doe": 32,
                "Bob Johnson": 28,
                "Alice Brown": 25
            },
            "total_inventory": 85,
            "avg_days_in_stock": 42,
            "aged_inventory_count": 18,
            "aged_inventory_pct": 0.21,
            "make_distribution": {
                "Toyota": 22,
                "Honda": 18,
                "Ford": 15,
                "Chevrolet": 12,
                "Nissan": 8,
                "BMW": 6,
                "Mercedes": 4
            },
            "new_used_distribution": {
                "New": 35,
                "Used": 50
            },
            "total_sessions": 2500,
            "total_users": 1800,
            "srp_to_vdp_ratio": 0.65,
            "page_views": {
                "/": 1200,
                "/inventory": 950,
                "/new-vehicles": 450,
                "/used-vehicles": 400,
                "/vehicle/honda-accord-a12345": 120,
                "/vehicle/toyota-camry-b67890": 110,
                "/vehicle/ford-f150-c24680": 105,
                "/finance": 350,
                "/service": 300,
                "/contact": 200
            },
            "top_pages": {
                "/": 1200,
                "/inventory": 950,
                "/new-vehicles": 450,
                "/used-vehicles": 400,
                "/vehicle/honda-accord-a12345": 120
            }
        },
        "previous_metrics": {
            "total_leads": 105,
            "lead_to_appointment": 0.28,
            "lead_to_sale": 0.12,
            "appointment_to_sale": 0.42,
            "total_inventory": 78,
            "avg_days_in_stock": 38,
            "aged_inventory_count": 14,
            "aged_inventory_pct": 0.18,
            "total_sessions": 2200,
            "total_users": 1600,
            "srp_to_vdp_ratio": 0.58
        },
        "anomalies": [
            {
                "metric": "lead_to_sale",
                "current_value": 0.15,
                "previous_value": 0.12,
                "change": 0.03,
                "percent_change": 0.25,
                "direction": "increased",
                "severity": "high"
            },
            {
                "metric": "srp_to_vdp_ratio",
                "current_value": 0.65,
                "previous_value": 0.58,
                "change": 0.07,
                "percent_change": 0.12,
                "direction": "increased",
                "severity": "medium"
            },
            {
                "metric": "aged_inventory_pct",
                "current_value": 0.21,
                "previous_value": 0.18,
                "change": 0.03,
                "percent_change": 0.17,
                "direction": "increased",
                "severity": "medium"
            }
        ],
        "text": """# Weekly Dealership Digest: 2023-04-01 to 2023-04-07

## Executive Summary
Your dealership shows strong performance in lead conversion metrics with a notable 25% increase in lead-to-sale ratio. Website traffic metrics are also improving, with increased SRP to VDP engagement. However, inventory aging metrics are trending negatively, with increasing average days in stock and aged inventory percentage.

## Lead Performance
Lead generation has increased by 14.3% week-over-week, with 120 total leads compared to 105 the previous week. More importantly, your lead conversion metrics show significant improvement:

* Lead-to-appointment ratio: 32% (up from 28%)
* Lead-to-sale ratio: 15% (up from 12%, a 25% improvement)
* Appointment-to-sale ratio: 45% (up from 42%)

Website leads remain your top source (37.5%), followed by phone leads (26.7%). Your top-performing sales rep is John Smith with 35 leads, followed closely by Jane Doe with 32 leads.

## Inventory Health
Your inventory levels have increased by 9% to 85 units, but inventory aging metrics show concerning trends:

* Average days in stock increased to 42 days (up from 38 days)
* Aged inventory (>60 days) increased to 21% of total inventory (up from 18%)

Toyota remains your top make (22 units), followed by Honda (18 units) and Ford (15 units). Your inventory mix is 41.2% new vehicles and 58.8% used vehicles.

## Website Traffic
Website engagement is showing positive trends with:

* 2,500 total sessions (up 13.6% from 2,200)
* 1,800 unique users (up 12.5% from 1,600)
* SRP to VDP ratio increased to 0.65 (up from 0.58)

Your homepage remains the most visited page, followed by the inventory search page. Vehicle detail pages for the Honda Accord, Toyota Camry, and Ford F-150 are your top-performing VDPs.

## Recommended Actions
1. Focus on inventory management strategies to address the increasing aged inventory percentage
2. Analyze the factors contributing to the improved lead-to-sale ratio and reinforce successful practices
3. Consider reallocating marketing spend toward website channels given their strong lead generation
4. Implement a specific strategy for aged inventory units (>60 days)""",
        "recommendations": [
            "Implement a 60-day inventory review process with automatic price adjustments for units approaching the 60-day threshold",
            "Document and share the current lead handling process that has led to the 25% increase in lead-to-sale conversion",
            "Increase digital marketing budget allocation to website channels by 15% based on their strong performance",
            "Create a weekend sales event specifically featuring vehicles that have been in inventory for 45+ days",
            "Analyze the Honda Accord VDP performance and apply its successful elements to other vehicle listing pages"
        ],
        "date_range": {
            "start": "2023-04-01",
            "end": "2023-04-07"
        }
    }
    
    render_insights_dashboard(demo_data)