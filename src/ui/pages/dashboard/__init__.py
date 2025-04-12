"""
Dashboard module for Watchdog AI.
"""
from .metrics import render_metrics_dashboard
from .insights import render_insights_dashboard
from src.ui.pages.dashboard.insights_engine import render_insights_engine_dashboard
from src.services.digest import DigestService
from src.services.storage import StorageService
from src.services.insight_engine import InsightEngine
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def render_dashboard_page():
    """Render the main dashboard page."""
    st.title("Dashboard")
    
    # Create tabs for different dashboard views
    tab1, tab2 = st.tabs(["Metrics", "Insights"])
    
    with tab1:
        render_metrics_dashboard()
    
    with tab2:
        render_insights_dashboard()

def render_overview_tab(digest_data, storage_service=None):
    """Render the overview tab with key metrics and charts."""
    st.subheader("Dealership Overview")
    
    # Extract metrics from digest data
    metrics = digest_data.get("metrics", {})
    previous_metrics = digest_data.get("previous_metrics", {})
    date_range = digest_data.get("date_range", {})
    
    # Show date range
    if date_range:
        start = date_range.get("start", "")
        end = date_range.get("end", "")
        if start and end:
            st.markdown(f"**Reporting Period:** {start} to {end}")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_leads = metrics.get("total_leads", 0)
        prev_leads = previous_metrics.get("total_leads", 0)
        delta = ((total_leads / prev_leads) - 1) * 100 if prev_leads else None
        delta_str = f"{delta:.1f}%" if delta is not None else None
        st.metric(label="Total Leads", value=total_leads, delta=delta_str)
    
    with col2:
        lead_to_sale = metrics.get("lead_to_sale", 0)
        prev_l2s = previous_metrics.get("lead_to_sale", 0)
        delta = (lead_to_sale - prev_l2s) * 100 if prev_l2s else None
        delta_str = f"{delta:.1f}pp" if delta is not None else None
        st.metric(label="Lead → Sale", value=f"{lead_to_sale * 100:.1f}%", delta=delta_str)
    
    with col3:
        days_in_stock = metrics.get("avg_days_in_stock", 0)
        prev_days = previous_metrics.get("avg_days_in_stock", 0)
        delta = days_in_stock - prev_days if prev_days else None
        delta_str = f"{delta:.1f}" if delta is not None else None
        st.metric(label="Avg Days in Stock", value=days_in_stock, delta=delta_str, delta_color="inverse")
    
    with col4:
        aged_pct = metrics.get("aged_inventory_pct", 0)
        prev_aged = previous_metrics.get("aged_inventory_pct", 0)
        delta = (aged_pct - prev_aged) * 100 if prev_aged else None
        delta_str = f"{delta:.1f}pp" if delta is not None else None
        st.metric(label="Aged Inventory", value=f"{aged_pct * 100:.1f}%", delta=delta_str, delta_color="inverse")
    
    # Create visualizations based on available data
    st.markdown("---")
    viz_col1, viz_col2 = st.columns(2)
    
    # Lead source chart
    with viz_col1:
        st.subheader("Lead Sources")
        lead_sources = metrics.get("lead_sources", {})
        if lead_sources:
            source_df = pd.DataFrame({
                "Source": list(lead_sources.keys()),
                "Count": list(lead_sources.values())
            })
            
            fig = px.pie(
                source_df, 
                values="Count", 
                names="Source",
                title="Lead Distribution by Source",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No lead source data available")
    
    # Inventory chart
    with viz_col2:
        st.subheader("Inventory by Make")
        make_distribution = metrics.get("make_distribution", {})
        if make_distribution:
            make_df = pd.DataFrame({
                "Make": list(make_distribution.keys()),
                "Count": list(make_distribution.values())
            }).sort_values("Count", ascending=False)
            
            fig = px.bar(
                make_df.head(8),
                x="Count",
                y="Make",
                orientation="h",
                title="Top Makes in Inventory",
                color="Count",
                color_continuous_scale=px.colors.sequential.Blues
            )
            fig.update_layout(yaxis=dict(categoryorder='total ascending'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inventory make data available")