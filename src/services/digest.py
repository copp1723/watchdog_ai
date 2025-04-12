"""
Digest generation service for Watchdog AI's Watchdog Mode.

This module handles the generation of weekly digests, anomaly detection, 
and scorecard generation.
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd

# Import Claude client
from src.ai.claude_client import ClaudeClient
from src.ai.prompts import get_digest_prompt, get_anomaly_detection_prompt, get_scorecard_prompt

# Import storage service
from src.services.storage import StorageService

# Import configuration
from config.config import DATA_DICTIONARY_PATH, EMAIL_CONFIG


class DigestService:
    """Service for generating digests, anomalies, and scorecards."""
    
    def __init__(self, storage_service: Optional[StorageService] = None, claude_client: Optional[ClaudeClient] = None):
        """
        Initialize the digest service.
        
        Args:
            storage_service: Optional StorageService (creates new one if not provided)
            claude_client: Optional ClaudeClient (creates new one if not provided)
        """
        self.storage_service = storage_service or StorageService()
        self.claude_client = claude_client or ClaudeClient()
        
        # Load data dictionary
        with open(DATA_DICTIONARY_PATH, 'r') as f:
            self.dictionary = json.load(f)
    
    def get_recent_data(self, days: int = 7) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Get recent data for digest generation.
        
        Args:
            days: Number of days to look back
            
        Returns:
            tuple: (success, data_or_error_message)
        """
        try:
            # List all parsed data
            success, files = self.storage_service.list_parsed_data()
            if not success:
                return False, files
            
            # Filter to recent files
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_files = []
            
            for file in files:
                # Extract timestamp from filename
                filename = file.get("name", "")
                if "_" in filename and filename.endswith(".json"):
                    try:
                        timestamp_str = filename.split("_")[-1].split(".")[0]
                        timestamp = int(timestamp_str)
                        file_date = datetime.fromtimestamp(timestamp)
                        
                        if file_date >= cutoff_date:
                            recent_files.append(file)
                    except (ValueError, IndexError):
                        # Skip files with invalid timestamps
                        pass
            
            # If no recent files, return error
            if not recent_files:
                return False, f"No data from the last {days} days found"
            
            # Load and combine data
            combined_data = {
                "leads": [],
                "inventory": [],
                "traffic": []
            }
            
            for file in recent_files:
                success, data = self.storage_service.get_parsed_data(file.get("name"))
                if success:
                    # Determine data type based on detected template
                    template = data.get("detected_template", "").lower()
                    file_data = data.get("data", [])
                    
                    if "lead" in template:
                        combined_data["leads"].extend(file_data)
                    elif "inventory" in template:
                        combined_data["inventory"].extend(file_data)
                    elif "ga4" in template or "traffic" in template:
                        combined_data["traffic"].extend(file_data)
            
            return True, combined_data
            
        except Exception as e:
            return False, f"Error getting recent data: {str(e)}"
    
    def calculate_metrics(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Calculate metrics for digest generation.
        
        Args:
            data: Combined data from get_recent_data
            
        Returns:
            dict: Calculated metrics
        """
        metrics = {}
        
        # Process leads data
        if data.get("leads"):
            leads_df = pd.DataFrame(data["leads"])
            
            # Convert date columns to datetime
            date_cols = [col for col in leads_df.columns if "date" in col.lower()]
            for col in date_cols:
                leads_df[col] = pd.to_datetime(leads_df[col], errors='coerce')
            
            # Total leads
            metrics["total_leads"] = len(leads_df)
            
            # Lead sources
            if "lead_source" in leads_df.columns:
                lead_sources = leads_df["lead_source"].value_counts().to_dict()
                metrics["lead_sources"] = lead_sources
                
                # Top lead sources
                top_sources = sorted(lead_sources.items(), key=lambda x: x[1], reverse=True)[:3]
                metrics["top_lead_sources"] = dict(top_sources)
            
            # Lead status
            if "status" in leads_df.columns:
                status_counts = leads_df["status"].value_counts().to_dict()
                metrics["lead_status"] = status_counts
                
                # Calculate conversion metrics
                if "Sold" in status_counts:
                    metrics["lead_to_sale"] = status_counts.get("Sold", 0) / len(leads_df)
                
                if "Appointment" in status_counts:
                    metrics["lead_to_appointment"] = status_counts.get("Appointment", 0) / len(leads_df)
                    
                    # If we have sold count, calculate appointment to sold
                    if "Sold" in status_counts:
                        appt_count = status_counts.get("Appointment", 0)
                        if appt_count > 0:
                            metrics["appointment_to_sale"] = status_counts.get("Sold", 0) / appt_count
            
            # Sales rep performance
            if "salesperson" in leads_df.columns or "salesrep" in leads_df.columns:
                rep_col = "salesperson" if "salesperson" in leads_df.columns else "salesrep"
                rep_performance = leads_df.groupby(rep_col).size().to_dict()
                metrics["rep_performance"] = rep_performance
        
        # Process inventory data
        if data.get("inventory"):
            inv_df = pd.DataFrame(data["inventory"])
            
            # Total inventory
            metrics["total_inventory"] = len(inv_df)
            
            # Days in stock
            if "days_in_stock" in inv_df.columns:
                metrics["avg_days_in_stock"] = inv_df["days_in_stock"].mean()
                
                # Aged inventory (>60 days)
                aged = inv_df[inv_df["days_in_stock"] > 60]
                metrics["aged_inventory_count"] = len(aged)
                metrics["aged_inventory_pct"] = len(aged) / len(inv_df) if len(inv_df) > 0 else 0
            
            # Make/model distribution
            if "make" in inv_df.columns:
                make_counts = inv_df["make"].value_counts().to_dict()
                metrics["make_distribution"] = make_counts
                
                # Top makes
                top_makes = sorted(make_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                metrics["top_makes"] = dict(top_makes)
            
            # New vs used
            if "new/used" in inv_df.columns or "new_used" in inv_df.columns:
                new_used_col = "new/used" if "new/used" in inv_df.columns else "new_used"
                new_used_counts = inv_df[new_used_col].value_counts().to_dict()
                metrics["new_used_distribution"] = new_used_counts
        
        # Process traffic data
        if data.get("traffic"):
            traffic_df = pd.DataFrame(data["traffic"])
            
            # Convert date columns to datetime
            if "date" in traffic_df.columns:
                traffic_df["date"] = pd.to_datetime(traffic_df["date"], errors='coerce')
            
            # Total sessions and users
            if "sessions" in traffic_df.columns:
                metrics["total_sessions"] = traffic_df["sessions"].sum()
            
            if "unique_users" in traffic_df.columns:
                metrics["total_users"] = traffic_df["unique_users"].sum()
            
            # Page performance
            if "page_path" in traffic_df.columns and "views" in traffic_df.columns:
                page_views = traffic_df.groupby("page_path")["views"].sum().to_dict()
                metrics["page_views"] = page_views
                
                # Top pages
                top_pages = sorted(page_views.items(), key=lambda x: x[1], reverse=True)[:5]
                metrics["top_pages"] = dict(top_pages)
            
            # Calculate SRP to VDP ratio
            if "page_path" in traffic_df.columns and "views" in traffic_df.columns:
                srp_views = traffic_df[traffic_df["page_path"] == "/inventory"]["views"].sum()
                vdp_views = traffic_df[traffic_df["page_path"].str.startswith("/vehicle/")]["views"].sum()
                
                if srp_views > 0:
                    metrics["srp_to_vdp_ratio"] = vdp_views / srp_views
        
        return metrics
    
    def detect_anomalies(self, current_metrics: Dict[str, Any], 
                        historical_metrics: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the metrics.
        
        Args:
            current_metrics: Current period metrics
            historical_metrics: Previous period metrics for comparison
            
        Returns:
            list: Detected anomalies
        """
        anomalies = []
        
        # If no historical data, we can't detect anomalies
        if not historical_metrics:
            return anomalies
        
        # Define threshold for significant changes
        thresholds = {
            "total_leads": 0.15,  # 15% change in lead volume
            "lead_to_sale": 0.05,  # 5 percentage point change in conversion
            "lead_to_appointment": 0.05,
            "appointment_to_sale": 0.05,
            "avg_days_in_stock": 7,  # 7 day change in average days in stock
            "aged_inventory_pct": 0.05,  # 5 percentage point change in aged inventory
            "total_sessions": 0.20,  # 20% change in website traffic
            "srp_to_vdp_ratio": 0.05  # 5 percentage point change in SRP:VDP ratio
        }
        
        # Check each metric for anomalies
        for metric, threshold in thresholds.items():
            if metric in current_metrics and metric in historical_metrics:
                current_val = current_metrics[metric]
                historical_val = historical_metrics[metric]
                
                # Skip if either value is not numeric
                if not isinstance(current_val, (int, float)) or not isinstance(historical_val, (int, float)):
                    continue
                
                # Calculate change
                change = current_val - historical_val
                pct_change = change / historical_val if historical_val != 0 else float('inf')
                
                # Check if change exceeds threshold
                if abs(pct_change) >= threshold:
                    direction = "increased" if change > 0 else "decreased"
                    
                    anomaly = {
                        "metric": metric,
                        "current_value": current_val,
                        "previous_value": historical_val,
                        "change": change,
                        "percent_change": pct_change,
                        "direction": direction,
                        "severity": "high" if abs(pct_change) >= threshold * 2 else "medium"
                    }
                    
                    anomalies.append(anomaly)
        
        # Sort anomalies by severity and percent change
        anomalies.sort(key=lambda x: (0 if x["severity"] == "high" else 1, -abs(x["percent_change"])))
        
        return anomalies
    
    def generate_scorecard(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a performance scorecard.
        
        Args:
            metrics: Calculated metrics
            
        Returns:
            dict: Scorecard with grades
        """
        scorecard = {}
        
        # Get industry benchmarks from data dictionary
        benchmarks = {}
        for metric_key, metric_data in self.dictionary.get("metrics", {}).items():
            benchmarks[metric_key] = metric_data.get("benchmark")
        
        # Grading function
        def grade_metric(value, benchmark):
            if benchmark is None:
                return "N/A"
            
            # Calculate percent difference from benchmark
            diff_pct = (value - benchmark) / benchmark
            
            # Assign grade based on difference
            if diff_pct > 0.20:  # >20% better than benchmark
                return "A"
            elif diff_pct > 0.05:  # 5-20% better
                return "B"
            elif diff_pct >= -0.05:  # Within 5% of benchmark
                return "C"
            elif diff_pct >= -0.20:  # 5-20% worse
                return "D"
            else:  # >20% worse
                return "F"
        
        # Grade each metric
        for metric_key, benchmark in benchmarks.items():
            # Convert from dictionary format to flat metrics format
            flat_key = metric_key.replace("_", "")
            
            if flat_key in metrics:
                value = metrics[flat_key]
                grade = grade_metric(value, benchmark)
                
                scorecard[metric_key] = {
                    "value": value,
                    "benchmark": benchmark,
                    "grade": grade,
                    "diff_pct": (value - benchmark) / benchmark if benchmark else None
                }
            elif metric_key in metrics:
                value = metrics[metric_key]
                grade = grade_metric(value, benchmark)
                
                scorecard[metric_key] = {
                    "value": value,
                    "benchmark": benchmark,
                    "grade": grade,
                    "diff_pct": (value - benchmark) / benchmark if benchmark else None
                }
        
        return scorecard
    
    def generate_weekly_digest(self) -> Tuple[bool, Union[str, Dict[str, Any]]]:
        """
        Generate a weekly digest with insights and anomalies.
        
        Returns:
            tuple: (success, digest_or_error_message)
        """
        try:
            # Get current week's data
            success, current_data = self.get_recent_data(days=7)
            if not success:
                return False, current_data
            
            # Calculate current metrics
            current_metrics = self.calculate_metrics(current_data)
            
            # Get previous week's data
            success, previous_data = self.get_recent_data(days=14)
            if success:
                # Filter to only include data from 8-14 days ago
                for key in previous_data:
                    if isinstance(previous_data[key], list):
                        previous_data[key] = [
                            item for item in previous_data[key] 
                            if "date" in item and 
                            datetime.fromisoformat(item["date"]) < datetime.now() - timedelta(days=7)
                        ]
                
                # Calculate previous metrics
                previous_metrics = self.calculate_metrics(previous_data)
                
                # Detect anomalies
                anomalies = self.detect_anomalies(current_metrics, previous_metrics)
            else:
                previous_metrics = {}
                anomalies = []
            
            # Generate scorecard
            scorecard = self.generate_scorecard(current_metrics)
            
            # Prepare data for digest
            digest_data = {
                "metrics": current_metrics,
                "previous_metrics": previous_metrics,
                "anomalies": anomalies,
                "scorecard": scorecard,
                "date_range": {
                    "start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "end": datetime.now().strftime("%Y-%m-%d")
                }
            }
            
            # Get prompts for Claude
            prompts = get_digest_prompt(digest_data)
            
            # Generate digest with Claude
            digest_text = self.claude_client.answer_question(
                prompts["user"], digest_data
            )
            
            # Return the digest
            result = {
                "text": digest_text,
                "data": digest_data,
                "generated_at": datetime.now().isoformat()
            }
            
            return True, result
            
        except Exception as e:
            return False, f"Error generating digest: {str(e)}"
    
    def send_digest_email(self, digest: Dict[str, Any], email: Optional[str] = None) -> Tuple[bool, str]:
        """
        Send the digest via email.
        
        Args:
            digest: The generated digest
            email: Recipient email (defaults to NOTIFICATION_EMAIL)
            
        Returns:
            tuple: (success, message)
        """
        # This will be implemented when SendGrid integration is added in Week 2
        # For now, just return a placeholder success message
        recipient = email or EMAIL_CONFIG.get("notification_email", "")
        return True, f"Email would be sent to {recipient} (functionality will be implemented in Week 2)"


if __name__ == "__main__":
    # Example usage
    digest_service = DigestService()
    success, result = digest_service.generate_weekly_digest()
    
    if success:
        print(result["text"])
    else:
        print(f"Error: {result}")