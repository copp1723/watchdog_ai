"""
Insight Generation Engine for Watchdog AI.

This module provides components for generating insights, detecting significant changes,
and performing comparative analysis on dealership metrics.
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
import json
import logging

# Import Claude client
from src.ai.claude_client import ClaudeClient
from src.ai.prompts import get_insight_prompt

# Import storage service
from src.services.storage import StorageService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightEngine:
    """Engine for generating dealership insights and detecting significant changes."""
    
    def __init__(self, storage_service: Optional[StorageService] = None, claude_client: Optional[ClaudeClient] = None):
        """
        Initialize the insight engine.
        
        Args:
            storage_service: Optional StorageService (creates new one if not provided)
            claude_client: Optional ClaudeClient (creates new one if not provided)
        """
        self.storage_service = storage_service or StorageService()
        self.claude_client = claude_client or ClaudeClient()
        
        # Define thresholds for significant changes
        self.significance_thresholds = {
            # Lead metrics
            "total_leads": 0.10,  # 10% change in lead volume
            "lead_to_appointment": 0.05,  # 5 percentage point change
            "lead_to_sale": 0.05,  # 5 percentage point change
            "appointment_to_sale": 0.05,  # 5 percentage point change
            
            # Inventory metrics
            "total_inventory": 0.15,  # 15% change in inventory
            "avg_days_in_stock": 7,  # 7 day change 
            "aged_inventory_pct": 0.05,  # 5 percentage point change
            
            # Traffic metrics
            "total_sessions": 0.15,  # 15% change in traffic
            "total_users": 0.15,  # 15% change in users
            "srp_to_vdp_ratio": 0.10,  # 10% change in SRP to VDP ratio
            
            # Default threshold for other metrics
            "default": 0.20  # 20% change for any other metric
        }
        
        # Define industry benchmarks for common dealership KPIs
        self.benchmarks = {
            "lead_to_appointment": 0.30,  # 30% of leads should convert to appointments
            "appointment_to_sale": 0.40,  # 40% of appointments should convert to sales
            "lead_to_sale": 0.12,  # 12% of leads should convert to sales
            "avg_days_in_stock": 35,  # Average days in stock benchmark
            "aged_inventory_pct": 0.15,  # Percentage of inventory over 60 days old
            "srp_to_vdp_ratio": 0.55,  # Ratio of VDPs viewed to SRPs viewed
            "bounce_rate": 0.40,  # Bounce rate benchmark
            "cost_per_lead": 25,  # Average cost per lead
        }
    
    def get_time_periods(self, days_in_period: int = 7, num_periods: int = 3) -> List[Dict[str, Any]]:
        """
        Get defined time periods for comparison.
        
        Args:
            days_in_period: Number of days in each period
            num_periods: Number of periods to return
            
        Returns:
            List of period definitions with start and end dates
        """
        today = datetime.now()
        periods = []
        
        for i in range(num_periods):
            end_date = today - timedelta(days=i * days_in_period)
            start_date = end_date - timedelta(days=days_in_period - 1)
            
            period = {
                "name": f"Period {i+1}",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "days": days_in_period,
                "is_current": i == 0
            }
            periods.append(period)
        
        return periods
    
    def get_data_for_period(self, period: Dict[str, Any]) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Get data for a specific time period.
        
        Args:
            period: Period definition with start and end dates
            
        Returns:
            tuple: (success, data_or_error_message)
        """
        try:
            # Convert string dates to datetime objects
            start_date = datetime.strptime(period["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(period["end_date"], "%Y-%m-%d")
            
            # Get all available parsed data
            success, files = self.storage_service.list_parsed_data()
            if not success:
                return False, files
            
            # Filter files to those within the date range
            period_files = []
            for file in files:
                # Extract timestamp from filename
                filename = file.get("name", "")
                if "_" in filename and filename.endswith(".json"):
                    try:
                        timestamp_str = filename.split("_")[-1].split(".")[0]
                        timestamp = int(timestamp_str)
                        file_date = datetime.fromtimestamp(timestamp)
                        
                        if start_date <= file_date <= end_date:
                            period_files.append(file)
                    except (ValueError, IndexError):
                        # Skip files with invalid timestamps
                        pass
            
            # If no files for this period, return error
            if not period_files:
                return False, f"No data found for period {period['start_date']} to {period['end_date']}"
            
            # Load and combine data
            combined_data = {
                "leads": [],
                "inventory": [],
                "traffic": []
            }
            
            for file in period_files:
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
            return False, f"Error getting data for period: {str(e)}"
    
    def calculate_metrics(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Calculate standard metrics for the provided data.
        
        Args:
            data: Combined data from get_data_for_period
            
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
                
                # Calculate rep efficiency if we have status information
                if "status" in leads_df.columns:
                    rep_efficiency = {}
                    for rep, count in rep_performance.items():
                        if count > 0:
                            # Sold count for this rep
                            sold_count = len(leads_df[(leads_df[rep_col] == rep) & (leads_df["status"] == "Sold")])
                            rep_efficiency[rep] = sold_count / count
                    
                    metrics["rep_efficiency"] = rep_efficiency
        
        # Process inventory data
        if data.get("inventory"):
            inv_df = pd.DataFrame(data["inventory"])
            
            # Total inventory
            metrics["total_inventory"] = len(inv_df)
            
            # Days in stock
            if "days_in_stock" in inv_df.columns:
                metrics["avg_days_in_stock"] = inv_df["days_in_stock"].mean()
                
                # Distribution of days in stock (for aging analysis)
                days_bins = [0, 30, 60, 90, float('inf')]
                days_labels = ['0-30', '31-60', '61-90', '90+']
                inv_df['age_group'] = pd.cut(inv_df['days_in_stock'], bins=days_bins, labels=days_labels)
                age_distribution = inv_df['age_group'].value_counts().to_dict()
                metrics["inventory_age_distribution"] = {str(k): v for k, v in age_distribution.items()}
                
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
                
                # If we have model column, get make-model combinations
                if "model" in inv_df.columns:
                    make_model_counts = inv_df.groupby(['make', 'model']).size().reset_index(name='count')
                    make_model_counts = make_model_counts.sort_values('count', ascending=False)
                    
                    # Get top 10 make-model combinations
                    top_make_models = {}
                    for _, row in make_model_counts.head(10).iterrows():
                        top_make_models[f"{row['make']} {row['model']}"] = row['count']
                    
                    metrics["top_make_models"] = top_make_models
            
            # New vs used
            if "new/used" in inv_df.columns or "new_used" in inv_df.columns:
                new_used_col = "new/used" if "new/used" in inv_df.columns else "new_used"
                new_used_counts = inv_df[new_used_col].value_counts().to_dict()
                metrics["new_used_distribution"] = new_used_counts
            
            # Price ranges if available
            price_cols = [col for col in inv_df.columns if "price" in col.lower()]
            if price_cols:
                price_col = price_cols[0]  # Use the first price column found
                
                # Convert to numeric, ignoring errors
                inv_df[price_col] = pd.to_numeric(inv_df[price_col], errors='coerce')
                
                # Calculate price metrics
                metrics["avg_price"] = inv_df[price_col].mean()
                metrics["median_price"] = inv_df[price_col].median()
                metrics["min_price"] = inv_df[price_col].min()
                metrics["max_price"] = inv_df[price_col].max()
                
                # Price distribution
                price_bins = [0, 10000, 20000, 30000, 40000, float('inf')]
                price_labels = ['<10k', '10k-20k', '20k-30k', '30k-40k', '40k+']
                inv_df['price_group'] = pd.cut(inv_df[price_col], bins=price_bins, labels=price_labels)
                price_distribution = inv_df['price_group'].value_counts().to_dict()
                metrics["price_distribution"] = {str(k): v for k, v in price_distribution.items()}
        
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
            
            if "bounce_rate" in traffic_df.columns:
                metrics["avg_bounce_rate"] = traffic_df["bounce_rate"].mean()
            
            if "session_duration" in traffic_df.columns:
                metrics["avg_session_duration"] = traffic_df["session_duration"].mean()
            
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
                    
                    # Also calculate SRP to lead ratio if we have lead counts
                    if "total_leads" in metrics and metrics["total_leads"] > 0:
                        metrics["srp_to_lead_ratio"] = srp_views / metrics["total_leads"]
            
            # Traffic sources if available
            if "source" in traffic_df.columns:
                source_counts = traffic_df.groupby("source")["sessions"].sum().to_dict()
                metrics["traffic_sources"] = source_counts
                
                # Top traffic sources
                top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                metrics["top_traffic_sources"] = dict(top_sources)
            
            # Device categories if available
            if "device_category" in traffic_df.columns:
                device_counts = traffic_df.groupby("device_category")["sessions"].sum().to_dict()
                metrics["device_categories"] = device_counts
        
        return metrics
    
    def detect_significant_changes(
        self, 
        current_metrics: Dict[str, Any], 
        previous_metrics: Dict[str, Any],
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect significant changes between two metric sets.
        
        Args:
            current_metrics: Current period metrics
            previous_metrics: Previous period metrics for comparison
            custom_thresholds: Optional custom thresholds for specific metrics
            
        Returns:
            list: Detected significant changes
        """
        significant_changes = []
        
        # Combine default thresholds with any custom thresholds
        thresholds = dict(self.significance_thresholds)
        if custom_thresholds:
            thresholds.update(custom_thresholds)
        
        # Iterate through current metrics and compare with previous
        for metric, current_value in current_metrics.items():
            # Skip if metric is not in previous data or is not numeric
            if (metric not in previous_metrics or 
                not isinstance(current_value, (int, float)) or 
                not isinstance(previous_metrics[metric], (int, float))):
                continue
            
            previous_value = previous_metrics[metric]
            
            # Skip if either value is 0 (would cause division by zero)
            if previous_value == 0:
                continue
            
            # Calculate absolute and percentage change
            absolute_change = current_value - previous_value
            percent_change = absolute_change / previous_value
            
            # Get threshold for this metric (use default if not specified)
            threshold = thresholds.get(metric, thresholds["default"])
            
            # Check if change exceeds threshold
            if abs(percent_change) >= threshold:
                # Determine if this is an improvement or decline
                is_improvement = self._is_metric_improvement(metric, absolute_change)
                
                # Determine severity based on how much it exceeds threshold
                severity = "high" if abs(percent_change) >= threshold * 2 else "medium"
                
                # Add to list of significant changes
                significant_changes.append({
                    "metric": metric,
                    "current_value": current_value,
                    "previous_value": previous_value,
                    "absolute_change": absolute_change,
                    "percent_change": percent_change,
                    "direction": "increased" if absolute_change > 0 else "decreased",
                    "is_improvement": is_improvement,
                    "severity": severity
                })
        
        # Sort by severity and absolute percentage change
        significant_changes.sort(key=lambda x: (
            0 if x["severity"] == "high" else 1,
            -abs(x["percent_change"])
        ))
        
        return significant_changes
    
    def _is_metric_improvement(self, metric: str, change: float) -> bool:
        """
        Determine if a change in a metric is an improvement.
        
        Args:
            metric: The metric name
            change: The absolute change in the metric
            
        Returns:
            bool: True if the change is an improvement, False otherwise
        """
        # Metrics where an increase is an improvement
        increase_good = [
            "total_leads", "lead_to_appointment", "lead_to_sale", 
            "appointment_to_sale", "total_sessions", "total_users",
            "srp_to_vdp_ratio", "rep_efficiency"
        ]
        
        # Metrics where a decrease is an improvement
        decrease_good = [
            "avg_days_in_stock", "aged_inventory_pct", "aged_inventory_count",
            "bounce_rate"
        ]
        
        if metric in increase_good:
            return change > 0
        elif metric in decrease_good:
            return change < 0
        else:
            # For any other metric, assume increase is good
            return change > 0
    
    def compare_to_benchmarks(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Compare metrics to industry benchmarks.
        
        Args:
            metrics: Current metrics
            
        Returns:
            dict: Benchmark comparison with grades
        """
        benchmark_comparison = {}
        
        # Compare each metric to benchmark if available
        for metric, value in metrics.items():
            if metric in self.benchmarks and isinstance(value, (int, float)):
                benchmark = self.benchmarks[metric]
                
                # Calculate difference from benchmark
                diff = value - benchmark
                diff_pct = diff / benchmark
                
                # Determine if this metric is better or worse than benchmark
                is_better = self._is_metric_improvement(metric, diff)
                
                # Assign grade based on percentage difference
                grade = self._assign_grade(diff_pct, is_better)
                
                benchmark_comparison[metric] = {
                    "value": value,
                    "benchmark": benchmark,
                    "diff": diff,
                    "diff_pct": diff_pct,
                    "is_better_than_benchmark": is_better,
                    "grade": grade
                }
        
        return benchmark_comparison
    
    def _assign_grade(self, diff_pct: float, is_improvement: bool) -> str:
        """
        Assign a letter grade based on percentage difference from benchmark.
        
        Args:
            diff_pct: Percentage difference from benchmark
            is_improvement: Whether this difference is an improvement
            
        Returns:
            str: Letter grade (A-F)
        """
        if not is_improvement:
            # Flip the percentage for metrics where lower is better
            diff_pct = -diff_pct
        
        # Grade scale
        if diff_pct >= 0.20:  # 20% or more better than benchmark
            return "A"
        elif diff_pct >= 0.05:  # 5-20% better
            return "B"
        elif diff_pct >= -0.05:  # Within 5% of benchmark
            return "C"
        elif diff_pct >= -0.20:  # 5-20% worse
            return "D"
        else:  # More than 20% worse
            return "F"
    
    def identify_trends(self, metrics_by_period: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Identify trends across multiple time periods.
        
        Args:
            metrics_by_period: List of metrics dictionaries, one per period
            
        Returns:
            dict: Identified trends for key metrics
        """
        trends = {}
        
        # Ensure we have at least 2 periods to analyze
        if len(metrics_by_period) < 2:
            return trends
        
        # Identify key metrics to track
        key_metrics = [
            "total_leads", "lead_to_sale", "lead_to_appointment", "appointment_to_sale",
            "total_inventory", "avg_days_in_stock", "aged_inventory_pct",
            "total_sessions", "total_users", "srp_to_vdp_ratio"
        ]
        
        # For each key metric, calculate trend
        for metric in key_metrics:
            # Extract values for this metric across periods
            values = []
            for period_metrics in metrics_by_period:
                if metric in period_metrics and isinstance(period_metrics[metric], (int, float)):
                    values.append(period_metrics[metric])
                else:
                    # If metric missing for any period, skip trend analysis
                    values = []
                    break
            
            # Skip if we don't have values for all periods
            if not values or len(values) != len(metrics_by_period):
                continue
            
            # Calculate trend direction
            # Simple method: compare most recent (first) to oldest (last)
            direction = "stable"
            if values[0] > values[-1] * 1.05:  # 5% increase
                direction = "increasing"
            elif values[0] < values[-1] * 0.95:  # 5% decrease
                direction = "decreasing"
            
            # Calculate consistency - are all periods moving in same direction?
            is_consistent = True
            for i in range(len(values) - 1):
                current_direction = "stable"
                if values[i] > values[i+1] * 1.02:  # 2% change between consecutive periods
                    current_direction = "increasing"
                elif values[i] < values[i+1] * 0.98:
                    current_direction = "decreasing"
                
                if current_direction != "stable" and current_direction != direction:
                    is_consistent = False
                    break
            
            # Calculate strength of trend (using line of best fit)
            if len(values) >= 3:
                # Simple linear regression
                x = np.array(range(len(values)))
                y = np.array(values)
                slope, _ = np.polyfit(x, y, 1)
                
                # Normalize slope by the mean of values
                mean_value = np.mean(values)
                normalized_slope = slope / mean_value if mean_value != 0 else 0
                
                # Interpret strength based on normalized slope
                strength = "weak"
                if abs(normalized_slope) >= 0.15:  # 15% change per period
                    strength = "strong"
                elif abs(normalized_slope) >= 0.05:  # 5% change per period
                    strength = "moderate"
            else:
                # Not enough data points for regression
                strength = "unknown"
            
            # Determine if trend is positive or negative
            is_positive = self._is_metric_improvement(metric, -1 if direction == "decreasing" else 1)
            
            # Add to trends dictionary
            trends[metric] = {
                "values": values,
                "direction": direction,
                "is_consistent": is_consistent,
                "strength": strength,
                "is_positive": is_positive
            }
        
        return trends
    
    def generate_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights using Claude and calculated metrics.
        
        Args:
            data: Dictionary with metrics, changes, and trends
            
        Returns:
            dict: Generated insights
        """
        try:
            # Get prompts for Claude
            prompts = get_insight_prompt(data)
            
            # Generate insights with Claude
            insight_text = self.claude_client.answer_question(
                prompts["user"], data
            )
            
            # Parse the insight text for structured data
            # Look for sections like ## Recommendations, ## Key Findings, etc.
            sections = {}
            
            # Extract sections
            section_pattern = r'##\s+(.*?)\n(.*?)(?=##|\Z)'
            import re
            matches = re.findall(section_pattern, insight_text, re.DOTALL)
            
            for title, content in matches:
                section_name = title.strip().lower().replace(' ', '_')
                sections[section_name] = content.strip()
            
            # Extract recommendations as bullet points
            recommendations = []
            if 'recommendations' in sections:
                rec_text = sections['recommendations']
                bullet_pattern = r'[\*\-]\s*(.*?)(?=[\*\-]|$)'
                rec_matches = re.findall(bullet_pattern, rec_text, re.DOTALL)
                recommendations = [match.strip() for match in rec_matches if match.strip()]
            
            # Return the insights
            return {
                "text": insight_text,
                "sections": sections,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {
                "error": str(e),
                "text": f"Error generating insights: {str(e)}",
                "sections": {},
                "recommendations": []
            }
    
    def complete_insight_analysis(self, days_in_period: int = 7, num_periods: int = 3) -> Dict[str, Any]:
        """
        Perform a complete insight analysis including metrics, changes, trends, and AI insights.
        
        Args:
            days_in_period: Days in each analysis period
            num_periods: Number of periods to analyze
            
        Returns:
            dict: Complete insight analysis
        """
        # Define time periods
        periods = self.get_time_periods(days_in_period, num_periods)
        
        # Calculate metrics for each period
        metrics_by_period = []
        for period in periods:
            success, data = self.get_data_for_period(period)
            if success:
                metrics = self.calculate_metrics(data)
                metrics["period"] = period
                metrics_by_period.append(metrics)
        
        # Ensure we have at least one period of data
        if not metrics_by_period:
            return {
                "error": "No data available for analysis",
                "periods": periods
            }
        
        # Get current and previous metrics
        current_metrics = metrics_by_period[0] if metrics_by_period else {}
        previous_metrics = metrics_by_period[1] if len(metrics_by_period) > 1 else {}
        
        # Detect significant changes
        significant_changes = []
        if previous_metrics:
            significant_changes = self.detect_significant_changes(current_metrics, previous_metrics)
        
        # Compare to benchmarks
        benchmark_comparison = self.compare_to_benchmarks(current_metrics)
        
        # Identify trends
        trends = self.identify_trends(metrics_by_period)
        
        # Prepare data for insight generation
        insight_data = {
            "metrics": current_metrics,
            "previous_metrics": previous_metrics,
            "significant_changes": significant_changes,
            "benchmark_comparison": benchmark_comparison,
            "trends": trends,
            "periods": periods
        }
        
        # Generate insights
        insights = self.generate_insights(insight_data)
        
        # Combine everything into a single result
        result = {
            "metrics": current_metrics,
            "previous_metrics": previous_metrics,
            "significant_changes": significant_changes,
            "benchmark_comparison": benchmark_comparison,
            "trends": trends,
            "insights": insights,
            "periods": periods,
            "generated_at": datetime.now().isoformat()
        }
        
        return result


if __name__ == "__main__":
    # Example usage
    engine = InsightEngine()
    result = engine.complete_insight_analysis()
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("=== Insight Analysis ===")
        print(f"Generated at: {result['generated_at']}")
        print(f"Periods analyzed: {len(result['periods'])}")
        print(f"Significant changes detected: {len(result['significant_changes'])}")
        print("\nTop insights:")
        if "insights" in result and "text" in result["insights"]:
            print(result["insights"]["text"][:500] + "...")