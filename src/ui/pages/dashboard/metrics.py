"""
Dashboard metrics visualization for Watchdog AI.

This module provides the metrics visualization components for the dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Color schemes
COLOR_SCHEME = {
    "primary": "#1F77B4",
    "secondary": "#FF7F0E",
    "success": "#2CA02C",
    "danger": "#D62728",
    "warning": "#BCBD22",
    "info": "#17BECF",
    "dark": "#7F7F7F",
    "mid": "#9467BD",
    "light": "#BBBBBB",
    "lead": "#1F77B4",
    "inventory": "#FF7F0E",
    "traffic": "#2CA02C",
    "conversion": "#9467BD"
}

def format_number(value: Union[int, float], is_percentage: bool = False) -> str:
    """Format a number for display"""
    if pd.isna(value):
        return "-"
    
    if is_percentage:
        return f"{value * 100:.1f}%" if isinstance(value, float) else f"{value}%"
    
    if isinstance(value, float):
        return f"{value:,.1f}"
    
    return f"{value:,}"

def format_time(seconds: Union[int, float]) -> str:
    """Format seconds as minutes:seconds"""
    if pd.isna(seconds):
        return "-"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

def metric_card(title: str, value: Union[int, float, str], 
               delta: Optional[Union[int, float]] = None,
               is_percentage: bool = False,
               is_time: bool = False,
               delta_color: str = "normal",
               help_text: Optional[str] = None):
    """Create a metric card with formatted value and optional delta"""
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Format the value based on type
        if is_time and not isinstance(value, str):
            formatted_value = format_time(value)
        elif is_percentage and not isinstance(value, str):
            formatted_value = format_number(value, is_percentage=True)
        elif isinstance(value, (int, float)):
            formatted_value = format_number(value)
        else:
            formatted_value = value
            
        # Create the metric
        if delta is not None:
            # Format delta
            if is_percentage:
                # For percentages, display absolute change in percentage points
                if isinstance(delta, float):
                    delta_value = f"{delta * 100:+.1f}pp"
                else:
                    delta_value = f"{delta:+}pp"
            else:
                # For regular numbers, display percentage change
                if isinstance(delta, float):
                    delta_value = f"{delta:+.1f}%"
                else:
                    delta_value = f"{delta:+}%"
                
            st.metric(
                title, 
                formatted_value, 
                delta=delta_value,
                delta_color=delta_color,
                help=help_text
            )
        else:
            st.metric(
                title, 
                formatted_value,
                help=help_text
            )
    
    return col2

def lead_metrics_section(metrics: Dict[str, Any], historical_metrics: Optional[Dict[str, Any]] = None):
    """Display lead metrics section"""
    st.subheader("Lead Metrics")
    
    # Extract metrics
    total_leads = metrics.get("total_leads", 0)
    lead_to_appointment = metrics.get("lead_to_appointment", 0)
    lead_to_sale = metrics.get("lead_to_sale", 0)
    appointment_to_sale = metrics.get("appointment_to_sale", 0)
    
    # Calculate deltas if historical data is available
    if historical_metrics:
        total_leads_delta = ((total_leads / historical_metrics.get("total_leads", 1)) - 1) * 100 if historical_metrics.get("total_leads") else None
        lead_to_appointment_delta = (lead_to_appointment - historical_metrics.get("lead_to_appointment", 0)) * 100 if "lead_to_appointment" in historical_metrics else None
        lead_to_sale_delta = (lead_to_sale - historical_metrics.get("lead_to_sale", 0)) * 100 if "lead_to_sale" in historical_metrics else None
        appointment_to_sale_delta = (appointment_to_sale - historical_metrics.get("appointment_to_sale", 0)) * 100 if "appointment_to_sale" in historical_metrics else None
    else:
        total_leads_delta = None
        lead_to_appointment_delta = None
        lead_to_sale_delta = None
        appointment_to_sale_delta = None
    
    # Create metrics row
    cols = st.columns(4)
    
    with cols[0]:
        metric_card(
            "Total Leads", 
            total_leads, 
            delta=total_leads_delta,
            help_text="Total number of leads received"
        )
    
    with cols[1]:
        metric_card(
            "Lead → Appointment", 
            lead_to_appointment,
            delta=lead_to_appointment_delta,
            is_percentage=True,
            help_text="Percentage of leads that converted to appointments"
        )
    
    with cols[2]:
        metric_card(
            "Lead → Sale", 
            lead_to_sale,
            delta=lead_to_sale_delta,
            is_percentage=True,
            help_text="Percentage of leads that converted to sales"
        )
    
    with cols[3]:
        metric_card(
            "Appointment → Sale", 
            appointment_to_sale,
            delta=appointment_to_sale_delta,
            is_percentage=True,
            help_text="Percentage of appointments that converted to sales"
        )
    
    # Source breakdown
    lead_sources = metrics.get("lead_sources", {})
    if lead_sources:
        st.write("##### Lead Sources")
        
        # Convert to dataframe for visualization
        source_df = pd.DataFrame({
            "Source": list(lead_sources.keys()),
            "Count": list(lead_sources.values())
        })
        source_df["Percentage"] = source_df["Count"] / source_df["Count"].sum()
        source_df = source_df.sort_values("Count", ascending=False)
        
        # Create columns for chart and table
        chart_col, table_col = st.columns([3, 2])
        
        with chart_col:
            # Create pie chart
            fig = px.pie(
                source_df, 
                values="Count", 
                names="Source",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                hole=0.4
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-.5,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with table_col:
            # Create formatted table
            display_df = source_df.copy()
            display_df["Percentage"] = display_df["Percentage"].apply(lambda x: f"{x:.1%}")
            st.dataframe(
                display_df, 
                hide_index=True,
                use_container_width=True
            )
    
    # Sales rep performance
    rep_performance = metrics.get("rep_performance", {})
    if rep_performance:
        st.write("##### Sales Rep Performance")
        
        # Convert to dataframe for visualization
        rep_df = pd.DataFrame({
            "Rep": list(rep_performance.keys()),
            "Leads": list(rep_performance.values())
        })
        rep_df = rep_df.sort_values("Leads", ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(
            rep_df,
            x="Leads",
            y="Rep",
            orientation="h",
            color="Leads",
            color_continuous_scale=px.colors.sequential.Blues,
            labels={"Leads": "Lead Count", "Rep": "Sales Rep"}
        )
        fig.update_layout(
            height=max(200, min(50 * len(rep_df), 500)),
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(categoryorder="total ascending")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add stats
        if len(rep_df) > 1:
            avg_leads = rep_df["Leads"].mean()
            max_leads = rep_df["Leads"].max()
            min_leads = rep_df["Leads"].min()
            
            stats_cols = st.columns(3)
            with stats_cols[0]:
                st.metric("Avg Leads per Rep", f"{avg_leads:.1f}")
            with stats_cols[1]:
                st.metric("Top Performer", f"{max_leads}")
            with stats_cols[2]:
                spread = max_leads - min_leads
                st.metric("Performance Spread", f"{spread}")

def inventory_metrics_section(metrics: Dict[str, Any], historical_metrics: Optional[Dict[str, Any]] = None):
    """Display inventory metrics section"""
    st.subheader("Inventory Metrics")
    
    # Extract metrics
    total_inventory = metrics.get("total_inventory", 0)
    avg_days = metrics.get("avg_days_in_stock", 0)
    aged_count = metrics.get("aged_inventory_count", 0)
    aged_pct = metrics.get("aged_inventory_pct", 0)
    
    # Calculate deltas if historical data is available
    if historical_metrics:
        total_inventory_delta = ((total_inventory / historical_metrics.get("total_inventory", 1)) - 1) * 100 if historical_metrics.get("total_inventory") else None
        avg_days_delta = avg_days - historical_metrics.get("avg_days_in_stock", 0) if "avg_days_in_stock" in historical_metrics else None
        aged_pct_delta = (aged_pct - historical_metrics.get("aged_inventory_pct", 0)) * 100 if "aged_inventory_pct" in historical_metrics else None
    else:
        total_inventory_delta = None
        avg_days_delta = None
        aged_pct_delta = None
    
    # Create metrics row
    cols = st.columns(3)
    
    with cols[0]:
        metric_card(
            "Total Inventory", 
            total_inventory, 
            delta=total_inventory_delta,
            help_text="Total units in inventory"
        )
    
    with cols[1]:
        days_delta_color = "inverse" if avg_days_delta is not None else "normal"
        metric_card(
            "Avg Days in Stock", 
            avg_days,
            delta=avg_days_delta,
            delta_color=days_delta_color,
            help_text="Average days vehicles have been in inventory"
        )
    
    with cols[2]:
        aged_delta_color = "inverse" if aged_pct_delta is not None else "normal"
        metric_card(
            "Aged Inventory", 
            aged_pct,
            delta=aged_pct_delta,
            is_percentage=True,
            delta_color=aged_delta_color,
            help_text="Percentage of inventory over 60 days old"
        )
    
    # Make/model breakdown
    make_distribution = metrics.get("make_distribution", {})
    if make_distribution:
        st.write("##### Inventory by Make")
        
        # Convert to dataframe for visualization
        make_df = pd.DataFrame({
            "Make": list(make_distribution.keys()),
            "Count": list(make_distribution.values())
        })
        make_df["Percentage"] = make_df["Count"] / make_df["Count"].sum()
        make_df = make_df.sort_values("Count", ascending=False)
        
        # Create columns for chart and table
        chart_col, table_col = st.columns([3, 2])
        
        with chart_col:
            # Create horizontal bar chart
            fig = px.bar(
                make_df.head(10),
                x="Count",
                y="Make",
                orientation="h",
                color="Count",
                color_continuous_scale=px.colors.sequential.Oranges,
                labels={"Count": "Units", "Make": "Make"}
            )
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(categoryorder="total ascending")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with table_col:
            # Create formatted table
            display_df = make_df.copy()
            display_df["Percentage"] = display_df["Percentage"].apply(lambda x: f"{x:.1%}")
            st.dataframe(
                display_df.head(10), 
                hide_index=True,
                use_container_width=True
            )
    
    # New vs Used breakdown
    new_used = metrics.get("new_used_distribution", {})
    if new_used:
        st.write("##### New vs. Used Inventory")
        
        # Convert to dataframe for visualization
        nu_df = pd.DataFrame({
            "Type": list(new_used.keys()),
            "Count": list(new_used.values())
        })
        
        # Create pie chart
        fig = px.pie(
            nu_df, 
            values="Count", 
            names="Type",
            color_discrete_sequence=[COLOR_SCHEME["primary"], COLOR_SCHEME["secondary"]],
            hole=0.4
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)

def traffic_metrics_section(metrics: Dict[str, Any], historical_metrics: Optional[Dict[str, Any]] = None):
    """Display traffic metrics section"""
    st.subheader("Website Traffic Metrics")
    
    # Extract metrics
    total_sessions = metrics.get("total_sessions", 0)
    total_users = metrics.get("total_users", 0)
    srp_to_vdp = metrics.get("srp_to_vdp_ratio", 0)
    
    # Calculate deltas if historical data is available
    if historical_metrics:
        total_sessions_delta = ((total_sessions / historical_metrics.get("total_sessions", 1)) - 1) * 100 if historical_metrics.get("total_sessions") else None
        total_users_delta = ((total_users / historical_metrics.get("total_users", 1)) - 1) * 100 if historical_metrics.get("total_users") else None
        srp_to_vdp_delta = (srp_to_vdp - historical_metrics.get("srp_to_vdp_ratio", 0)) * 100 if "srp_to_vdp_ratio" in historical_metrics else None
    else:
        total_sessions_delta = None
        total_users_delta = None
        srp_to_vdp_delta = None
    
    # Create metrics row
    cols = st.columns(3)
    
    with cols[0]:
        metric_card(
            "Total Sessions", 
            total_sessions, 
            delta=total_sessions_delta,
            help_text="Total website sessions"
        )
    
    with cols[1]:
        metric_card(
            "Unique Users", 
            total_users,
            delta=total_users_delta,
            help_text="Unique website visitors"
        )
    
    with cols[2]:
        metric_card(
            "SRP to VDP Ratio", 
            srp_to_vdp,
            delta=srp_to_vdp_delta,
            help_text="Ratio of vehicle detail page views to inventory search views"
        )
    
    # Top pages
    top_pages = metrics.get("top_pages", {})
    if top_pages:
        st.write("##### Top Pages by Views")
        
        # Convert to dataframe for visualization
        pages_df = pd.DataFrame({
            "Page": list(top_pages.keys()),
            "Views": list(top_pages.values())
        })
        pages_df = pages_df.sort_values("Views", ascending=False)
        
        # Clean up page paths for display
        pages_df["Display Page"] = pages_df["Page"].apply(lambda x: x.replace("/vehicle/", "/vehicle/ ") if x.startswith("/vehicle/") else x)
        
        # Create horizontal bar chart
        fig = px.bar(
            pages_df.head(10),
            x="Views",
            y="Display Page",
            orientation="h",
            color="Views",
            color_continuous_scale=px.colors.sequential.Greens,
            labels={"Views": "Page Views", "Display Page": "Page"}
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(categoryorder="total ascending")
        )
        st.plotly_chart(fig, use_container_width=True)

def anomalies_section(anomalies: List[Dict[str, Any]]):
    """Display anomalies section"""
    if not anomalies:
        return
    
    st.subheader("Detected Anomalies")
    
    # Group anomalies by severity
    high_anomalies = [a for a in anomalies if a.get("severity") == "high"]
    medium_anomalies = [a for a in anomalies if a.get("severity") == "medium"]
    
    if high_anomalies:
        st.markdown("##### Critical Changes")
        for anomaly in high_anomalies:
            with st.container(border=True):
                metric = anomaly.get("metric", "").replace("_", " ").title()
                direction = anomaly.get("direction", "")
                change = anomaly.get("percent_change", 0)
                current = anomaly.get("current_value", 0)
                previous = anomaly.get("previous_value", 0)
                
                if abs(change) > 1:
                    change_text = f"{abs(change):.1f}x"
                else:
                    change_text = f"{abs(change) * 100:.1f}%"
                
                st.markdown(f"**{metric}** has {direction} by {change_text}")
                
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Current:** {current}")
                with cols[1]:
                    st.markdown(f"**Previous:** {previous}")
    
    if medium_anomalies:
        st.markdown("##### Notable Changes")
        for anomaly in medium_anomalies:
            with st.container(border=True):
                metric = anomaly.get("metric", "").replace("_", " ").title()
                direction = anomaly.get("direction", "")
                change = anomaly.get("percent_change", 0)
                current = anomaly.get("current_value", 0)
                previous = anomaly.get("previous_value", 0)
                
                if abs(change) > 1:
                    change_text = f"{abs(change):.1f}x"
                else:
                    change_text = f"{abs(change) * 100:.1f}%"
                
                st.markdown(f"**{metric}** has {direction} by {change_text}")
                
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Current:** {current}")
                with cols[1]:
                    st.markdown(f"**Previous:** {previous}")

def scorecard_section(scorecard: Dict[str, Any]):
    """Display scorecard section"""
    if not scorecard:
        return
    
    st.subheader("Performance Scorecard")
    
    # Group metrics by grade
    metrics_by_grade = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "F": []
    }
    
    for metric, data in scorecard.items():
        grade = data.get("grade", "")
        if grade in metrics_by_grade:
            display_metric = metric.replace("_", " ").title()
            metrics_by_grade[grade].append({
                "metric": display_metric,
                "value": data.get("value"),
                "benchmark": data.get("benchmark"),
                "diff_pct": data.get("diff_pct")
            })
    
    # Create columns for different grade groups
    col1, col2 = st.columns(2)
    
    with col1:
        # Top performers
        st.markdown("##### Top Performers (A/B)")
        for metric in metrics_by_grade["A"] + metrics_by_grade["B"]:
            with st.container(border=True):
                m_name = metric["metric"]
                value = metric["value"]
                benchmark = metric["benchmark"]
                diff = metric["diff_pct"]
                
                if diff is not None:
                    diff_text = f"{diff * 100:.1f}% better than benchmark"
                else:
                    diff_text = "No benchmark data"
                
                st.markdown(f"**{m_name}:** {value} (Benchmark: {benchmark})")
                st.markdown(f"_{diff_text}_")
    
    with col2:
        # Needs improvement
        st.markdown("##### Needs Improvement (D/F)")
        if metrics_by_grade["D"] + metrics_by_grade["F"]:
            for metric in metrics_by_grade["D"] + metrics_by_grade["F"]:
                with st.container(border=True):
                    m_name = metric["metric"]
                    value = metric["value"]
                    benchmark = metric["benchmark"]
                    diff = metric["diff_pct"]
                    
                    if diff is not None:
                        diff_text = f"{abs(diff) * 100:.1f}% worse than benchmark"
                    else:
                        diff_text = "No benchmark data"
                    
                    st.markdown(f"**{m_name}:** {value} (Benchmark: {benchmark})")
                    st.markdown(f"_{diff_text}_")
        else:
            st.info("No metrics performing below benchmark")
    
    # Show average metrics
    st.markdown("##### Meet Expectations (C)")
    if metrics_by_grade["C"]:
        c_metrics = [f"**{m['metric']}**" for m in metrics_by_grade["C"]]
        st.markdown(", ".join(c_metrics))
    else:
        st.info("No metrics at benchmark level")

def render_metrics_dashboard(digest_data: Dict[str, Any]):
    """Render the metrics dashboard"""
    st.title("Watchdog AI Dashboard")
    
    # Get metrics from digest data
    metrics = digest_data.get("metrics", {})
    previous_metrics = digest_data.get("previous_metrics", {})
    anomalies = digest_data.get("anomalies", [])
    scorecard = digest_data.get("scorecard", {})
    date_range = digest_data.get("date_range", {})
    
    # Show date range
    if date_range:
        start = date_range.get("start", "")
        end = date_range.get("end", "")
        if start and end:
            st.markdown(f"**Reporting Period:** {start} to {end}")
    
    # Tabs for different metric categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Lead Metrics", "Inventory Metrics", "Traffic Metrics", "Scorecard"
    ])
    
    with tab1:
        # Show anomalies and key metrics
        if anomalies:
            anomalies_section(anomalies)
        else:
            st.info("No significant changes detected")
        
        st.divider()
        
        # Show key metrics from each category
        st.subheader("Key Metrics")
        
        key_metrics_cols = st.columns(3)
        
        with key_metrics_cols[0]:
            st.markdown("##### Lead Performance")
            metric_card(
                "Total Leads", 
                metrics.get("total_leads", 0),
                delta=((metrics.get("total_leads", 0) / previous_metrics.get("total_leads", 1)) - 1) * 100 if previous_metrics.get("total_leads") else None
            )
            
            metric_card(
                "Lead → Sale", 
                metrics.get("lead_to_sale", 0),
                delta=(metrics.get("lead_to_sale", 0) - previous_metrics.get("lead_to_sale", 0)) * 100 if "lead_to_sale" in previous_metrics else None,
                is_percentage=True
            )
        
        with key_metrics_cols[1]:
            st.markdown("##### Inventory Health")
            metric_card(
                "Total Inventory", 
                metrics.get("total_inventory", 0),
                delta=((metrics.get("total_inventory", 0) / previous_metrics.get("total_inventory", 1)) - 1) * 100 if previous_metrics.get("total_inventory") else None
            )
            
            aged_delta_color = "inverse"
            metric_card(
                "Aged Inventory", 
                metrics.get("aged_inventory_pct", 0),
                delta=(metrics.get("aged_inventory_pct", 0) - previous_metrics.get("aged_inventory_pct", 0)) * 100 if "aged_inventory_pct" in previous_metrics else None,
                is_percentage=True,
                delta_color=aged_delta_color
            )
        
        with key_metrics_cols[2]:
            st.markdown("##### Website Traffic")
            metric_card(
                "Total Sessions", 
                metrics.get("total_sessions", 0),
                delta=((metrics.get("total_sessions", 0) / previous_metrics.get("total_sessions", 1)) - 1) * 100 if previous_metrics.get("total_sessions") else None
            )
            
            metric_card(
                "SRP to VDP Ratio", 
                metrics.get("srp_to_vdp_ratio", 0),
                delta=(metrics.get("srp_to_vdp_ratio", 0) - previous_metrics.get("srp_to_vdp_ratio", 0)) * 100 if "srp_to_vdp_ratio" in previous_metrics else None
            )
    
    with tab2:
        lead_metrics_section(metrics, previous_metrics)
    
    with tab3:
        inventory_metrics_section(metrics, previous_metrics)
    
    with tab4:
        traffic_metrics_section(metrics, previous_metrics)
    
    with tab5:
        scorecard_section(scorecard)


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
            }
        ],
        "scorecard": {
            "lead_to_sale": {
                "value": 0.15,
                "benchmark": 0.12,
                "grade": "B",
                "diff_pct": 0.25
            },
            "appointment_to_sale": {
                "value": 0.45,
                "benchmark": 0.40,
                "grade": "B",
                "diff_pct": 0.125
            },
            "avg_days_in_stock": {
                "value": 42,
                "benchmark": 35,
                "grade": "D",
                "diff_pct": -0.2
            },
            "aged_inventory_pct": {
                "value": 0.21,
                "benchmark": 0.15,
                "grade": "F",
                "diff_pct": -0.4
            },
            "srp_to_vdp_ratio": {
                "value": 0.65,
                "benchmark": 0.55,
                "grade": "A",
                "diff_pct": 0.18
            }
        },
        "date_range": {
            "start": "2023-04-01",
            "end": "2023-04-07"
        }
    }
    
    render_metrics_dashboard(demo_data)