"""
GA4 traffic reports parser for Watchdog AI.

This module handles parsing and processing Google Analytics 4 traffic reports.
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from ..template_mapper import TemplateMapper


class GA4Parser:
    """Parser for Google Analytics 4 traffic reports."""
    
    def __init__(self, mapper: Optional[TemplateMapper] = None):
        """
        Initialize the GA4 parser.
        
        Args:
            mapper: Optional TemplateMapper instance for standardization
        """
        self.mapper = mapper
        self.required_columns = [
            "Date", "Page Path", "Sessions", "Unique Users", "Views"
        ]
        self.metrics_columns = [
            "Sessions", "Unique Users", "Views", "Bounce Rate", "Exit Rate"
        ]
    
    def parse_file(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse a GA4 traffic report file.
        
        Args:
            file_path: Path to the GA4 report file (CSV)
            **kwargs: Additional arguments to pass to pandas read_csv
            
        Returns:
            Tuple of (processed_dataframe, processing_results)
        """
        # Check if file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path, **kwargs)
        
        # Validate that this is a GA4 report
        if not self._is_ga4_report(df):
            raise ValueError("This file does not appear to be a valid GA4 traffic report")
        
        # Standardize and process data
        processed_df, results = self._process_ga4_data(df, file_path)
        
        return processed_df, results
    
    def _is_ga4_report(self, df: pd.DataFrame) -> bool:
        """
        Check if a dataframe is a valid GA4 traffic report.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Boolean indicating if this is a GA4 report
        """
        # Check for required columns
        for col in self.required_columns:
            if col not in df.columns:
                return False
        
        # Check if the format of the data matches GA4 expectations
        # Date column should contain dates
        try:
            pd.to_datetime(df["Date"], errors="raise")
        except:
            return False
        
        # Page Path should generally start with "/"
        if not df["Page Path"].str.startswith("/").any():
            return False
        
        return True
    
    def _process_ga4_data(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process and standardize GA4 data.
        
        Args:
            df: GA4 report DataFrame
            file_path: Path to the source file
            
        Returns:
            Tuple of (processed_dataframe, processing_results)
        """
        # Initial basic processing of metrics
        processed_df = df.copy()
        
        # Convert date to datetime
        processed_df["Date"] = pd.to_datetime(processed_df["Date"])
        
        # Convert percentage strings to floats
        for col in ["Bounce Rate", "Exit Rate"]:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].str.rstrip("%").astype(float) / 100
        
        # Convert time on page to seconds
        if "Avg. Time on Page" in processed_df.columns:
            processed_df["Time on Page (seconds)"] = processed_df["Avg. Time on Page"].apply(self._time_to_seconds)
        
        # Calculate additional metrics
        processed_df["Views per Session"] = processed_df["Views"] / processed_df["Sessions"]
        
        # Create page type categories
        processed_df["Page Type"] = processed_df["Page Path"].apply(self._categorize_page)
        
        # Prepare results
        results = {
            "file_name": Path(file_path).name,
            "row_count": len(df),
            "date_range": {
                "start": processed_df["Date"].min().strftime("%Y-%m-%d"),
                "end": processed_df["Date"].max().strftime("%Y-%m-%d"),
                "days": (processed_df["Date"].max() - processed_df["Date"].min()).days + 1
            },
            "metrics_summary": self._calculate_metrics_summary(processed_df),
            "page_types": processed_df["Page Type"].value_counts().to_dict()
        }
        
        return processed_df, results
    
    def _time_to_seconds(self, time_str: str) -> int:
        """
        Convert a time string (HH:MM:SS) to seconds.
        
        Args:
            time_str: Time string in format HH:MM:SS
            
        Returns:
            Total seconds
        """
        try:
            parts = time_str.split(":")
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            elif len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            else:
                return 0
        except:
            return 0
    
    def _categorize_page(self, path: str) -> str:
        """
        Categorize a page path into a page type.
        
        Args:
            path: Page path
            
        Returns:
            Page type category
        """
        if path == "/" or path == "/home" or path == "/index":
            return "Homepage"
        elif path.startswith("/vehicle/"):
            return "VDP"  # Vehicle Detail Page
        elif path == "/inventory" or path.startswith("/inventory/"):
            return "SRP"  # Search Results Page
        elif path.startswith("/new-") or path.startswith("/used-"):
            return "Category Page"
        elif path == "/contact" or path.startswith("/contact/"):
            return "Contact Page"
        elif path == "/service" or path.startswith("/service/"):
            return "Service Page"
        elif path == "/finance" or path.startswith("/finance/"):
            return "Finance Page"
        elif path == "/specials" or path.startswith("/special") or path.startswith("/offer"):
            return "Specials Page"
        elif path == "/about" or path.startswith("/about/"):
            return "About Page"
        else:
            return "Other"
    
    def _calculate_metrics_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary metrics for the GA4 data.
        
        Args:
            df: Processed GA4 dataframe
            
        Returns:
            Dictionary of summary metrics
        """
        # Group by date and calculate totals
        daily_metrics = df.groupby("Date").agg({
            "Sessions": "sum",
            "Unique Users": "sum",
            "Views": "sum",
            "Entrances": "sum"
        }).reset_index()
        
        # Calculate high-level metrics
        total_sessions = df["Sessions"].sum()
        total_users = df["Unique Users"].sum()
        total_pageviews = df["Views"].sum()
        avg_pages_per_session = total_pageviews / total_sessions if total_sessions > 0 else 0
        
        # Calculate bounce and exit rates
        bounce_rate = None
        exit_rate = None
        if "Bounce Rate" in df.columns and "Entrances" in df.columns:
            weighted_bounces = (df["Bounce Rate"] * df["Entrances"]).sum()
            bounce_rate = weighted_bounces / df["Entrances"].sum() if df["Entrances"].sum() > 0 else None
        
        if "Exit Rate" in df.columns and "Views" in df.columns:
            weighted_exits = (df["Exit Rate"] * df["Views"]).sum()
            exit_rate = weighted_exits / df["Views"].sum() if df["Views"].sum() > 0 else None
        
        # Calculate time metrics
        avg_time_on_page = None
        if "Time on Page (seconds)" in df.columns and "Views" in df.columns:
            avg_time_on_page = (df["Time on Page (seconds)"] * df["Views"]).sum() / df["Views"].sum() if df["Views"].sum() > 0 else None
        
        # Calculate traffic by page type
        traffic_by_page_type = df.groupby("Page Type").agg({
            "Sessions": "sum",
            "Views": "sum"
        }).reset_index()
        
        traffic_by_page_type["Sessions_pct"] = traffic_by_page_type["Sessions"] / total_sessions if total_sessions > 0 else 0
        traffic_by_page_type["Views_pct"] = traffic_by_page_type["Views"] / total_pageviews if total_pageviews > 0 else 0
        
        # Return all summary metrics
        return {
            "total_sessions": int(total_sessions),
            "total_users": int(total_users),
            "total_pageviews": int(total_pageviews),
            "avg_pages_per_session": float(avg_pages_per_session),
            "bounce_rate": float(bounce_rate) if bounce_rate is not None else None,
            "exit_rate": float(exit_rate) if exit_rate is not None else None,
            "avg_time_on_page_seconds": float(avg_time_on_page) if avg_time_on_page is not None else None,
            "traffic_by_page_type": traffic_by_page_type.to_dict(orient="records")
        }
    
    def generate_insights(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate insights from GA4 data.
        
        Args:
            df: Processed GA4 dataframe
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        # Top pages by traffic
        top_pages = df.groupby(["Page Path", "Page Title"]).agg({
            "Sessions": "sum",
            "Views": "sum"
        }).reset_index().sort_values("Views", ascending=False).head(10)
        
        insights.append({
            "type": "top_pages",
            "title": "Top Pages by Traffic",
            "data": top_pages.to_dict(orient="records")
        })
        
        # Pages with high bounce rates
        if "Bounce Rate" in df.columns and "Entrances" in df.columns:
            high_bounce_pages = df[df["Entrances"] > 10].sort_values("Bounce Rate", ascending=False).head(10)
            
            insights.append({
                "type": "high_bounce_pages",
                "title": "Pages with High Bounce Rates",
                "data": high_bounce_pages[["Page Path", "Page Title", "Bounce Rate", "Entrances"]].to_dict(orient="records")
            })
        
        # Pages with high engagement
        if "Time on Page (seconds)" in df.columns:
            high_engagement_pages = df[df["Views"] > 10].sort_values("Time on Page (seconds)", ascending=False).head(10)
            
            insights.append({
                "type": "high_engagement_pages",
                "title": "Pages with Highest Engagement",
                "data": high_engagement_pages[["Page Path", "Page Title", "Time on Page (seconds)", "Views"]].to_dict(orient="records")
            })
        
        # Daily trend analysis
        daily_metrics = df.groupby("Date").agg({
            "Sessions": "sum",
            "Unique Users": "sum", 
            "Views": "sum"
        }).reset_index()
        
        insights.append({
            "type": "daily_trends",
            "title": "Daily Traffic Trends",
            "data": daily_metrics.to_dict(orient="records")
        })
        
        return insights
    
    def get_vdp_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get performance metrics for Vehicle Detail Pages (VDPs).
        
        Args:
            df: Processed GA4 dataframe
            
        Returns:
            DataFrame with VDP performance metrics
        """
        # Filter to VDP pages
        vdp_df = df[df["Page Path"].str.startswith("/vehicle/")].copy()
        
        # Extract vehicle info from the page path
        vdp_df["Vehicle"] = vdp_df["Page Title"].str.split("Stock #").str[0].str.strip()
        vdp_df["Stock Number"] = vdp_df["Page Path"].str.split("-").str[-1]
        
        # Aggregate metrics by vehicle
        vdp_metrics = vdp_df.groupby(["Vehicle", "Stock Number"]).agg({
            "Sessions": "sum",
            "Views": "sum",
            "Time on Page (seconds)": "mean",
            "Bounce Rate": "mean"
        }).reset_index()
        
        return vdp_metrics


if __name__ == "__main__":
    # Sample usage
    parser = GA4Parser()
    file_path = "assets/sample_data/ga4_traffic_report.csv"
    
    try:
        df, results = parser.parse_file(file_path)
        print(f"Processed {results['row_count']} rows of traffic data")
        print(f"Date range: {results['date_range']['start']} to {results['date_range']['end']} ({results['date_range']['days']} days)")
        print(f"Total sessions: {results['metrics_summary']['total_sessions']}")
        print(f"Total pageviews: {results['metrics_summary']['total_pageviews']}")
        
        # Generate insights
        insights = parser.generate_insights(df)
        print(f"Generated {len(insights)} insights")
        
        # Get VDP performance
        vdp_metrics = parser.get_vdp_performance(df)
        print(f"Found {len(vdp_metrics)} vehicles with VDP visits")
        
    except Exception as e:
        print(f"Error processing GA4 report: {str(e)}")