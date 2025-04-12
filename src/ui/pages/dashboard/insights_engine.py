"""
Dashboard insights engine visualization for Watchdog AI.

This module provides the advanced insights engine visualization for the dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union
import re
from datetime import datetime, timedelta

# Import insight engine
from src.services.insight_engine import InsightEngine

# Import Claude client for generating insights on demand
from src.ai.claude_client import ClaudeClient


def format_metric_value(metric: str, value: Union[int, float]) -> str:
    """Format metric value for display with proper units"""
    if isinstance(value, float) and (metric.endswith("_pct") or "_to_" in metric or "ratio" in metric):
        return f"{value * 100:.1f}%"
    elif isinstance(value, float):
        return f"{value:.2f}"
    else:
        return f"{value:,}"


def render_trends_section(trends: Dict[str, Dict[str, Any]]):
    """Render the trends visualization section"""
    st.subheader("Performance Trends")
    
    if not trends:
        st.info("Insufficient data to identify trends. More historical data is needed.")
        return
    
    # Group trends by direction and significance
    improving = []
    declining = []
    stable = []
    
    for metric, data in trends.items():
        direction = data.get("direction", "stable")
        is_positive = data.get("is_positive", False)
        strength = data.get("strength", "weak")
        values = data.get("values", [])
        
        # Format the metric name for display
        display_name = metric.replace("_", " ").title()
        
        trend_info = {
            "metric": display_name,
            "raw_metric": metric,
            "direction": direction,
            "is_positive": is_positive,
            "strength": strength,
            "values": values
        }
        
        if direction == "stable":
            stable.append(trend_info)
        elif (direction == "increasing" and is_positive) or (direction == "decreasing" and not is_positive):
            improving.append(trend_info)
        else:
            declining.append(trend_info)
    
    # Sort by strength
    def sort_by_strength(x):
        strength_order = {"strong": 0, "moderate": 1, "weak": 2, "unknown": 3}
        return strength_order.get(x["strength"], 4)
    
    improving.sort(key=sort_by_strength)
    declining.sort(key=sort_by_strength)
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Improving Metrics")
        if improving:
            for trend in improving:
                with st.container(border=True):
                    metric = trend["metric"]
                    strength = trend["strength"].title()
                    direction = trend["direction"].title()
                    values = trend["values"]
                    
                    # Format the values for display
                    formatted_values = [format_metric_value(trend["raw_metric"], v) for v in values]
                    
                    st.markdown(f"**{metric}**")
                    st.markdown(f"_{strength} {direction} Trend_")
                    
                    # Create sparkline chart
                    if len(values) >= 2:
                        periods = [f"Period {i+1}" for i in range(len(values))]
                        periods.reverse()  # Reverse to show most recent on right
                        values_to_plot = values.copy()
                        values_to_plot.reverse()  # Reverse to match periods
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=periods,
                            y=values_to_plot,
                            mode='lines+markers',
                            line=dict(color='green', width=2),
                            marker=dict(color='green', size=6)
                        ))
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=100,
                            showlegend=False,
                            xaxis=dict(showticklabels=False),
                            yaxis=dict(showticklabels=False)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show current vs oldest
                        st.markdown(f"Current: **{formatted_values[0]}** → Previous: {formatted_values[-1]}")
        else:
            st.info("No improving metrics detected.")
    
    with col2:
        st.markdown("#### Declining Metrics")
        if declining:
            for trend in declining:
                with st.container(border=True):
                    metric = trend["metric"]
                    strength = trend["strength"].title()
                    direction = trend["direction"].title()
                    values = trend["values"]
                    
                    # Format the values for display
                    formatted_values = [format_metric_value(trend["raw_metric"], v) for v in values]
                    
                    st.markdown(f"**{metric}**")
                    st.markdown(f"_{strength} {direction} Trend_")
                    
                    # Create sparkline chart
                    if len(values) >= 2:
                        periods = [f"Period {i+1}" for i in range(len(values))]
                        periods.reverse()  # Reverse to show most recent on right
                        values_to_plot = values.copy()
                        values_to_plot.reverse()  # Reverse to match periods
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=periods,
                            y=values_to_plot,
                            mode='lines+markers',
                            line=dict(color='red', width=2),
                            marker=dict(color='red', size=6)
                        ))
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=100,
                            showlegend=False,
                            xaxis=dict(showticklabels=False),
                            yaxis=dict(showticklabels=False)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show current vs oldest
                        st.markdown(f"Current: **{formatted_values[0]}** → Previous: {formatted_values[-1]}")
        else:
            st.info("No declining metrics detected.")
    
    # Show stable metrics if any
    if stable:
        st.markdown("#### Stable Metrics")
        stable_metrics = [s["metric"] for s in stable]
        st.markdown(", ".join(stable_metrics))


def render_benchmark_section(benchmark_comparison: Dict[str, Dict[str, Any]]):
    """Render benchmark comparison visualization"""
    st.subheader("Benchmark Comparison")
    
    if not benchmark_comparison:
        st.info("No benchmark data available for comparison.")
        return
    
    # Group metrics by grade
    metrics_by_grade = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "F": []
    }
    
    for metric, data in benchmark_comparison.items():
        grade = data.get("grade", "")
        if grade in metrics_by_grade:
            # Format the metric name for display
            display_name = metric.replace("_", " ").title()
            
            # Format values
            value = format_metric_value(metric, data["value"])
            benchmark = format_metric_value(metric, data["benchmark"])
            
            # Calculate percentage difference
            diff_pct = data.get("diff_pct", 0)
            if diff_pct > 0:
                diff_display = f"{diff_pct * 100:.1f}% better"
            elif diff_pct < 0:
                diff_display = f"{abs(diff_pct) * 100:.1f}% worse"
            else:
                diff_display = "at benchmark"
            
            metrics_by_grade[grade].append({
                "metric": display_name,
                "value": value,
                "benchmark": benchmark,
                "diff_display": diff_display,
                "diff_pct": diff_pct
            })
    
    # Create tabs for each grade group
    tab1, tab2, tab3 = st.tabs([
        "🌟 Excellent (A/B)", "⚖️ At Benchmark (C)", "⚠️ Needs Improvement (D/F)"
    ])
    
    with tab1:
        excellent = metrics_by_grade["A"] + metrics_by_grade["B"]
        if excellent:
            columns = st.columns(2)
            for i, metric in enumerate(excellent):
                col = columns[i % 2]
                with col:
                    with st.container(border=True):
                        st.markdown(f"### {metric['metric']}")
                        st.markdown(f"**Current:** {metric['value']}")
                        st.markdown(f"**Benchmark:** {metric['benchmark']}")
                        st.markdown(f"_{metric['diff_display']} than benchmark_")
                        
                        # Add a simple progress bar
                        progress_val = min(1.0, max(0.0, 0.5 + metric['diff_pct'] / 2))
                        st.progress(progress_val, text=None)
        else:
            st.info("No metrics exceeding benchmarks.")
    
    with tab2:
        at_benchmark = metrics_by_grade["C"]
        if at_benchmark:
            columns = st.columns(2)
            for i, metric in enumerate(at_benchmark):
                col = columns[i % 2]
                with col:
                    with st.container(border=True):
                        st.markdown(f"### {metric['metric']}")
                        st.markdown(f"**Current:** {metric['value']}")
                        st.markdown(f"**Benchmark:** {metric['benchmark']}")
                        st.markdown(f"_{metric['diff_display']} than benchmark_")
        else:
            st.info("No metrics at benchmark level.")
    
    with tab3:
        needs_improvement = metrics_by_grade["D"] + metrics_by_grade["F"]
        if needs_improvement:
            columns = st.columns(2)
            for i, metric in enumerate(needs_improvement):
                col = columns[i % 2]
                with col:
                    with st.container(border=True):
                        st.markdown(f"### {metric['metric']}")
                        st.markdown(f"**Current:** {metric['value']}")
                        st.markdown(f"**Benchmark:** {metric['benchmark']}")
                        st.markdown(f"_{metric['diff_display']} than benchmark_")
                        
                        # Add a simple progress bar
                        progress_val = min(1.0, max(0.0, 0.5 + metric['diff_pct'] / 2))
                        st.progress(progress_val, text=None)
        else:
            st.info("No metrics below benchmark level.")


def render_significant_changes(changes: List[Dict[str, Any]]):
    """Render significant changes visualization"""
    st.subheader("Significant Changes")
    
    if not changes:
        st.info("No significant changes detected.")
        return
    
    # Group by severity
    high_severity = [c for c in changes if c.get("severity") == "high"]
    medium_severity = [c for c in changes if c.get("severity") == "medium"]
    
    # Create tabs for different severity levels
    tab1, tab2 = st.tabs([
        f"Critical Changes ({len(high_severity)})",
        f"Notable Changes ({len(medium_severity)})"
    ])
    
    with tab1:
        if high_severity:
            for change in high_severity:
                metric = change.get("metric", "").replace("_", " ").title()
                current = change.get("current_value", 0)
                previous = change.get("previous_value", 0)
                percent_change = change.get("percent_change", 0)
                direction = change.get("direction", "")
                is_improvement = change.get("is_improvement", False)
                
                # Format for display
                if isinstance(current, float) and (metric.lower().endswith("pct") or "to" in metric.lower() or "ratio" in metric.lower()):
                    current_display = f"{current * 100:.1f}%"
                    previous_display = f"{previous * 100:.1f}%"
                else:
                    current_display = f"{current:,}"
                    previous_display = f"{previous:,}"
                
                # Format percent change
                if abs(percent_change) >= 1:
                    pct_display = f"{abs(percent_change):.1f}x"
                else:
                    pct_display = f"{abs(percent_change) * 100:.1f}%"
                
                # Determine color based on whether this is an improvement
                color = "green" if is_improvement else "red"
                
                with st.container(border=True):
                    st.markdown(f"### {metric}")
                    st.markdown(f"<span style='color:{color};font-size:1.2em;font-weight:bold;'>{direction.title()} by {pct_display}</span>", unsafe_allow_html=True)
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Current", current_display)
                    with cols[1]:
                        st.metric("Previous", previous_display)
        else:
            st.info("No critical changes detected.")
    
    with tab2:
        if medium_severity:
            columns = st.columns(2)
            for i, change in enumerate(medium_severity):
                col = columns[i % 2]
                
                with col:
                    metric = change.get("metric", "").replace("_", " ").title()
                    current = change.get("current_value", 0)
                    previous = change.get("previous_value", 0)
                    percent_change = change.get("percent_change", 0)
                    direction = change.get("direction", "")
                    is_improvement = change.get("is_improvement", False)
                    
                    # Format for display
                    if isinstance(current, float) and (metric.lower().endswith("pct") or "to" in metric.lower() or "ratio" in metric.lower()):
                        current_display = f"{current * 100:.1f}%"
                        previous_display = f"{previous * 100:.1f}%"
                    else:
                        current_display = f"{current:,}"
                        previous_display = f"{previous:,}"
                    
                    # Format percent change
                    if abs(percent_change) >= 1:
                        pct_display = f"{abs(percent_change):.1f}x"
                    else:
                        pct_display = f"{abs(percent_change) * 100:.1f}%"
                    
                    # Determine color based on whether this is an improvement
                    color = "green" if is_improvement else "red"
                    
                    with st.container(border=True):
                        st.markdown(f"#### {metric}")
                        st.markdown(f"<span style='color:{color};'>{direction.title()} by {pct_display}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Current:** {current_display}")
                        st.markdown(f"**Previous:** {previous_display}")
        else:
            st.info("No notable changes detected.")


def render_ai_insights(insights: Dict[str, Any], engine: InsightEngine):
    """Render AI-generated insights section"""
    st.subheader("AI-Powered Insights")
    
    # Check if we have insights text
    insights_text = insights.get("text", "")
    sections = insights.get("sections", {})
    recommendations = insights.get("recommendations", [])
    
    if not insights_text:
        st.warning("No AI insights available. Click 'Generate Insights' to create a new analysis.")
        
        if st.button("Generate Insights"):
            with st.spinner("Generating AI-powered insights..."):
                try:
                    # This would get the complete analysis, but for demo we'll fake it
                    # insights = engine.generate_insights(st.session_state.insight_data)
                    
                    # For demo purposes:
                    insights = {
                        "text": """# Dealership Performance Insights

## Executive Summary
The dealership is showing strong performance in lead conversion metrics with a notable 25% increase in lead-to-sale ratio, now at 15% compared to 12% previously. However, inventory health metrics are concerning, with aged inventory percentage increasing to 21% and receiving an F grade compared to the 15% industry benchmark.

## Key Findings
* Lead-to-sale ratio has improved significantly to 15%, earning a B grade (25% above benchmark)
* SRP to VDP ratio shows strong engagement at 0.65, earning an A grade (18% above benchmark)
* Aged inventory has increased to 21% of total inventory, earning an F grade (40% worse than benchmark)
* Average days in stock has increased to 42 days, earning a D grade (20% worse than benchmark)
* Total leads have increased by 14.3% to 120 leads

## Improvement Opportunities
* Inventory aging is a critical area needing attention with aged inventory percentage (21%) significantly higher than the industry benchmark (15%)
* Average days in stock (42 days) is trending in the wrong direction and now 20% worse than benchmark
* While lead volume has increased, there may be opportunity to improve lead source diversity, with 37.5% coming from website

## Strengths
* Lead conversion performance is excellent, with lead-to-sale ratio (15%) exceeding benchmark by 25%
* Website engagement is strong with SRP to VDP ratio (0.65) exceeding benchmark by 18%
* Appointment-to-sale ratio (45%) is performing above benchmark by 12.5%
* Lead volume is showing healthy growth with a 14.3% increase

## Recommendations
* Implement a 30/60/90 day inventory management strategy with automatic price adjustments at each threshold to address the increasing aged inventory issue
* Create a focused sales event specifically for vehicles approaching 60 days in inventory to reduce the aged inventory percentage
* Document and formalize the lead handling process that has delivered the 25% improvement in lead-to-sale ratio to ensure consistent application
* Analyze the factors contributing to the improved SRP to VDP ratio and apply those insights to other areas of the website
* Consider increasing marketing investment in non-website lead sources to diversify lead generation
* Conduct a detailed analysis of inventory turn rates by make/model to identify specific inventory categories causing the increase in average days in stock
* Implement a weekly inventory aging report for management review to maintain focus on this critical area""",
                        "sections": {
                            "executive_summary": "The dealership is showing strong performance in lead conversion metrics with a notable 25% increase in lead-to-sale ratio, now at 15% compared to 12% previously. However, inventory health metrics are concerning, with aged inventory percentage increasing to 21% and receiving an F grade compared to the 15% industry benchmark.",
                            "key_findings": "* Lead-to-sale ratio has improved significantly to 15%, earning a B grade (25% above benchmark)\n* SRP to VDP ratio shows strong engagement at 0.65, earning an A grade (18% above benchmark)\n* Aged inventory has increased to 21% of total inventory, earning an F grade (40% worse than benchmark)\n* Average days in stock has increased to 42 days, earning a D grade (20% worse than benchmark)\n* Total leads have increased by 14.3% to 120 leads",
                            "improvement_opportunities": "* Inventory aging is a critical area needing attention with aged inventory percentage (21%) significantly higher than the industry benchmark (15%)\n* Average days in stock (42 days) is trending in the wrong direction and now 20% worse than benchmark\n* While lead volume has increased, there may be opportunity to improve lead source diversity, with 37.5% coming from website",
                            "strengths": "* Lead conversion performance is excellent, with lead-to-sale ratio (15%) exceeding benchmark by 25%\n* Website engagement is strong with SRP to VDP ratio (0.65) exceeding benchmark by 18%\n* Appointment-to-sale ratio (45%) is performing above benchmark by 12.5%\n* Lead volume is showing healthy growth with a 14.3% increase",
                            "recommendations": "* Implement a 30/60/90 day inventory management strategy with automatic price adjustments at each threshold to address the increasing aged inventory issue\n* Create a focused sales event specifically for vehicles approaching 60 days in inventory to reduce the aged inventory percentage\n* Document and formalize the lead handling process that has delivered the 25% improvement in lead-to-sale ratio to ensure consistent application\n* Analyze the factors contributing to the improved SRP to VDP ratio and apply those insights to other areas of the website\n* Consider increasing marketing investment in non-website lead sources to diversify lead generation\n* Conduct a detailed analysis of inventory turn rates by make/model to identify specific inventory categories causing the increase in average days in stock\n* Implement a weekly inventory aging report for management review to maintain focus on this critical area"
                        },
                        "recommendations": [
                            "Implement a 30/60/90 day inventory management strategy with automatic price adjustments at each threshold to address the increasing aged inventory issue",
                            "Create a focused sales event specifically for vehicles approaching 60 days in inventory to reduce the aged inventory percentage",
                            "Document and formalize the lead handling process that has delivered the 25% improvement in lead-to-sale ratio to ensure consistent application",
                            "Analyze the factors contributing to the improved SRP to VDP ratio and apply those insights to other areas of the website",
                            "Consider increasing marketing investment in non-website lead sources to diversify lead generation",
                            "Conduct a detailed analysis of inventory turn rates by make/model to identify specific inventory categories causing the increase in average days in stock",
                            "Implement a weekly inventory aging report for management review to maintain focus on this critical area"
                        ]
                    }
                    
                    # Store in session state
                    st.session_state.insights = insights
                    
                    # Force a rerun to display the new insights
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
        
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Executive Summary", "Key Findings", "Strengths & Opportunities", "Recommendations"
    ])
    
    with tab1:
        if "executive_summary" in sections:
            st.markdown(sections["executive_summary"])
        else:
            st.markdown(insights_text.split("##")[0])  # Default to first section
    
    with tab2:
        if "key_findings" in sections:
            st.markdown(sections["key_findings"])
        else:
            st.info("No key findings section available in the insights.")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Strengths")
            if "strengths" in sections:
                st.markdown(sections["strengths"])
            else:
                st.info("No strengths section available.")
        
        with col2:
            st.markdown("### Improvement Opportunities")
            if "improvement_opportunities" in sections:
                st.markdown(sections["improvement_opportunities"])
            else:
                st.info("No improvement opportunities section available.")
    
    with tab4:
        if recommendations:
            for i, rec in enumerate(recommendations):
                with st.container(border=True):
                    st.markdown(f"#### Recommendation {i+1}")
                    st.markdown(rec)
        elif "recommendations" in sections:
            st.markdown(sections["recommendations"])
        else:
            st.info("No recommendations available.")
    
    # Add a button to regenerate insights
    if st.button("Regenerate Insights"):
        st.session_state.insights = None
        st.experimental_rerun()


def render_insights_engine_dashboard(insight_data: Optional[Dict[str, Any]] = None):
    """Render the complete insights engine dashboard"""
    st.title("Insight Generation Engine")
    st.markdown('<p class="subtitle">AI-powered analysis and recommendations</p>', unsafe_allow_html=True)
    
    # Initialize insight engine
    try:
        insight_engine = InsightEngine()
    except Exception as e:
        st.error(f"Error initializing insight engine: {str(e)}")
        insight_engine = None
        return
    
    # Use provided data or generate demo data
    if not insight_data:
        # For demonstration, use the same sample data as in digest.py
        insight_data = {
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
            "significant_changes": [
                {
                    "metric": "lead_to_sale",
                    "current_value": 0.15,
                    "previous_value": 0.12,
                    "absolute_change": 0.03,
                    "percent_change": 0.25,
                    "direction": "increased",
                    "is_improvement": True,
                    "severity": "high"
                },
                {
                    "metric": "srp_to_vdp_ratio",
                    "current_value": 0.65,
                    "previous_value": 0.58,
                    "absolute_change": 0.07,
                    "percent_change": 0.12,
                    "direction": "increased",
                    "is_improvement": True,
                    "severity": "medium"
                },
                {
                    "metric": "aged_inventory_pct",
                    "current_value": 0.21,
                    "previous_value": 0.18,
                    "absolute_change": 0.03,
                    "percent_change": 0.17,
                    "direction": "increased",
                    "is_improvement": False,
                    "severity": "medium"
                }
            ],
            "benchmark_comparison": {
                "lead_to_sale": {
                    "value": 0.15,
                    "benchmark": 0.12,
                    "diff": 0.03,
                    "diff_pct": 0.25,
                    "is_better_than_benchmark": True,
                    "grade": "B"
                },
                "appointment_to_sale": {
                    "value": 0.45,
                    "benchmark": 0.40,
                    "diff": 0.05,
                    "diff_pct": 0.125,
                    "is_better_than_benchmark": True,
                    "grade": "B"
                },
                "avg_days_in_stock": {
                    "value": 42,
                    "benchmark": 35,
                    "diff": 7,
                    "diff_pct": -0.2,
                    "is_better_than_benchmark": False,
                    "grade": "D"
                },
                "aged_inventory_pct": {
                    "value": 0.21,
                    "benchmark": 0.15,
                    "diff": 0.06,
                    "diff_pct": -0.4,
                    "is_better_than_benchmark": False,
                    "grade": "F"
                },
                "srp_to_vdp_ratio": {
                    "value": 0.65,
                    "benchmark": 0.55,
                    "diff": 0.1,
                    "diff_pct": 0.18,
                    "is_better_than_benchmark": True,
                    "grade": "A"
                }
            },
            "trends": {
                "total_leads": {
                    "values": [120, 105, 90],
                    "direction": "increasing",
                    "is_consistent": True,
                    "strength": "moderate",
                    "is_positive": True
                },
                "lead_to_sale": {
                    "values": [0.15, 0.12, 0.11],
                    "direction": "increasing",
                    "is_consistent": True,
                    "strength": "strong",
                    "is_positive": True
                },
                "avg_days_in_stock": {
                    "values": [42, 38, 32],
                    "direction": "increasing",
                    "is_consistent": True,
                    "strength": "moderate",
                    "is_positive": False
                },
                "aged_inventory_pct": {
                    "values": [0.21, 0.18, 0.15],
                    "direction": "increasing",
                    "is_consistent": True,
                    "strength": "moderate",
                    "is_positive": False
                },
                "srp_to_vdp_ratio": {
                    "values": [0.65, 0.58, 0.55],
                    "direction": "increasing",
                    "is_consistent": True,
                    "strength": "moderate",
                    "is_positive": True
                }
            },
            "periods": [
                {
                    "name": "Period 1",
                    "start_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "end_date": datetime.now().strftime("%Y-%m-%d"),
                    "days": 7,
                    "is_current": True
                },
                {
                    "name": "Period 2",
                    "start_date": (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"),
                    "end_date": (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d"),
                    "days": 7,
                    "is_current": False
                },
                {
                    "name": "Period 3",
                    "start_date": (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d"),
                    "end_date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
                    "days": 7,
                    "is_current": False
                }
            ]
        }
        
        # Store in session state for reuse
        st.session_state.insight_data = insight_data
    
    # Check if we have insights in session state
    if "insights" not in st.session_state:
        st.session_state.insights = {
            "text": "",
            "sections": {},
            "recommendations": []
        }
    
    # Period dates for display
    periods = insight_data.get("periods", [])
    if periods:
        current_period = next((p for p in periods if p.get("is_current", False)), periods[0])
        st.markdown(f"**Analysis Period:** {current_period.get('start_date')} to {current_period.get('end_date')}")
    
    # Create tabs for different insight sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "AI Insights", "Significant Changes", "Performance Trends", "Benchmark Comparison"
    ])
    
    with tab1:
        render_ai_insights(st.session_state.insights, insight_engine)
    
    with tab2:
        render_significant_changes(insight_data.get("significant_changes", []))
    
    with tab3:
        render_trends_section(insight_data.get("trends", {}))
    
    with tab4:
        render_benchmark_section(insight_data.get("benchmark_comparison", {}))


if __name__ == "__main__":
    # Demo for individual component testing
    render_insights_engine_dashboard()