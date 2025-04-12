"""
Prompts for Claude API interactions in Watchdog AI.

This module contains prompt templates for various Claude interactions.
"""
from typing import Dict, List, Any, Optional
import json


def get_data_analysis_prompt(summary: Dict[str, Any], samples: Dict[str, List[Any]]) -> str:
    """
    Generate a prompt for data analysis
    
    Args:
        summary: Data summary information
        samples: Sample values for each field
        
    Returns:
        Formatted prompt
    """
    prompt = f"""You are an expert in dealership data analysis. Below is a summary of a dataset from a car dealership, along with sample values. Please analyze this data and provide insights.

## DATA SUMMARY
- Row count: {summary.get('row_count', 'unknown')}
- Column count: {summary.get('column_count', 'unknown')}
- Detected template: {summary.get('detected_template', 'unknown')}
- Template confidence: {summary.get('template_confidence', 0) * 100:.1f}%

## COLUMNS
{', '.join(summary.get('columns', []))}

## SAMPLE VALUES
"""

    # Add sample values for each column
    for col, values in samples.items():
        value_str = ', '.join([str(v) for v in values[:5]])
        prompt += f"- {col}: {value_str}\n"

    prompt += """
## REQUESTED ANALYSIS
1. DATA TYPE IDENTIFICATION: What type of dealership data is this (leads, inventory, sales, etc.)?
2. DATA QUALITY ASSESSMENT: Are there any missing values, inconsistent formats, or potential errors?
3. KEY INSIGHTS: What are the most important patterns or insights from this data?
4. RECOMMENDATIONS: What actions should be taken to improve or utilize this data?

Please provide your analysis in a structured format with clear sections. When discussing data quality issues, be specific about which columns have problems and what the issues are."""

    return prompt


def get_data_validation_prompt(data: Dict[str, Any], sample_rows: int = 5) -> Dict[str, str]:
    """
    Get prompts for data validation.
    
    Args:
        data: The parsed data containing summary and samples
        sample_rows: Number of sample rows to include
        
    Returns:
        dict: System and user prompts for data validation
    """
    # Prepare the data sample
    data_sample = data.get("sample_data", [])[:sample_rows] if "sample_data" in data else []
    columns = data.get("columns", [])
    
    system_prompt = """You are an automotive dealership data expert helping to validate data for Watchdog AI.
Analyze the provided data summary and sample data to identify potential issues or insights.
Focus on data quality, completeness, and potential anomalies.

For dealership data, be alert for:
1. Missing key fields (dates, lead sources, salesperson names)
2. Outliers in numeric fields (extremely high/low values)
3. Data consistency issues (future dates, impossible values)
4. Formatting problems that might affect analysis

Respond in JSON format with a comprehensive assessment."""
    
    user_prompt = f"""Please validate this dealership data and provide insights.

Data Summary:
- Filename: {data.get('filename', 'Unknown')}
- Row count: {data.get('row_count', 0)}
- Template type: {data.get('detected_template', 'Unknown')}
- Template confidence: {data.get('template_confidence', 0):.2f}
- Available columns: {', '.join(columns)}

Column Mapping:
{data.get('mapped_columns', {})}

Sample Data (first {sample_rows} rows):
{data_sample}

Please analyze this data and answer:
1. Is this data valid and usable for dealership analysis?
2. Are there any issues or anomalies you notice?
3. What kind of insights could we derive from this data?
4. Any recommendations for improving this data?

Format your response as JSON with these keys:
{{
    "is_valid": true/false,
    "issues": ["issue1", "issue2", ...],
    "potential_insights": ["insight1", "insight2", ...],
    "recommendations": ["rec1", "rec2", ...]
}}"""
    
    return {
        "system": system_prompt,
        "user": user_prompt
    }


def get_column_mapping_prompt(columns: List[str], detected_template: str = None) -> str:
    """
    Generate a prompt for column mapping suggestions
    
    Args:
        columns: List of column names
        detected_template: Optional detected template name
        
    Returns:
        Formatted prompt
    """
    template_context = f"I've detected this may be a {detected_template} format. " if detected_template else ""
    
    prompt = f"""You are an expert in dealership data structures. {template_context}I need to map these column names to standardized field names.

## COLUMN NAMES
{', '.join(columns)}

## STANDARD FIELD CATEGORIES
- date_fields: date, lead_date, sale_date, close_date
- lead_source_fields: lead_source, source, traffic_source, lead_type
- salesperson_fields: salesperson, sales_rep, agent, employee
- inventory_fields: stock_number, vin, days_in_stock
- vehicle_fields: year, make, model, trim, color, ext_color, int_color
- customer_fields: customer_name, first_name, last_name, email, phone
- pricing_fields: price, msrp, selling_price, cost, profit

## TASK
For each column name, suggest the most appropriate standardized field name.
Return your suggestions as a JSON object mapping original column names to standardized field names.

Example format:
```json
{
  "CUST_NAME": "customer_name",
  "VEH_YR": "year",
  "SALE_PRICE": "selling_price"
}
```"""

    return prompt


def get_chat_prompt(question: str, context: dict = None) -> str:
    """
    Get the prompt for chat interactions.
    
    Args:
        question: User's question
        context: Optional data context
        
    Returns:
        str: Formatted prompt for chat interactions
    """
    base_prompt = f"Question: {question}\n\n"
    
    if context:
        base_prompt += f"Context: {str(context)}\n\n"
    
    base_prompt += "Please provide a clear and concise answer based on the available information."
    
    return base_prompt


def get_data_validation_prompt(data_summary: dict) -> str:
    """
    Get the prompt for data validation.
    
    Args:
        data_summary: Summary of the data to validate
        
    Returns:
        str: Formatted prompt
    """
    return f"""Please analyze this dealership data:
{str(data_summary)}

Identify any potential data quality issues or anomalies."""


def get_digest_prompt(data: Dict[str, Any], timeframe: str = "week") -> Dict[str, str]:
    """
    Get prompts for digest generation.
    
    Args:
        data: The data to analyze
        timeframe: Timeframe for the digest (day, week, month)
        
    Returns:
        dict: System and user prompts
    """
    system_prompt = f"""
    You are an automotive dealership data expert creating a {timeframe}ly digest for Watchdog AI.
    Analyze the provided data and highlight key insights, trends, and anomalies.
    Focus on actionable information that dealership managers can use to improve performance.
    
    Your digest should follow this structure:
    1. Executive Summary (2-3 sentences on overall performance)
    2. Key Metrics (3-5 most important metrics with context)
    3. Notable Trends (what's changing)
    4. Anomalies (any outliers or unusual patterns)
    5. Action Items (2-3 specific recommendations)
    
    Use clear, concise language and focus on business impact.
    """
    
    # Format the datasets into a digestible format
    datasets_summary = []
    for name, dataset in data.items():
        dataset_info = f"Dataset: {name}\n"
        dataset_info += f"- Rows: {dataset.get('row_count', 0)}\n"
        dataset_info += f"- Type: {dataset.get('detected_template', 'Unknown')}\n"
        
        # Add metrics if available
        if "metrics" in dataset:
            dataset_info += "- Metrics:\n"
            for metric_name, metric_value in dataset["metrics"].items():
                if not isinstance(metric_value, dict) and not isinstance(metric_value, list):
                    dataset_info += f"  - {metric_name}: {metric_value}\n"
        
        datasets_summary.append(dataset_info)
    
    user_prompt = f"""
    Please generate a {timeframe}ly digest for the dealership based on these datasets:
    
    {chr(10).join(datasets_summary)}
    
    Focus on actionable insights and notable patterns. What should the dealership manager pay attention to?
    """
    
    return {
        "system": system_prompt,
        "user": user_prompt
    }


def get_anomaly_detection_prompt(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Get prompts for anomaly detection.
    
    Args:
        data: The data to analyze for anomalies
        
    Returns:
        dict: System and user prompts
    """
    system_prompt = """
    You are an automotive dealership data expert specializing in anomaly detection.
    Your task is to analyze the provided dealership metrics and identify significant
    changes or anomalies that could require management attention.
    
    Focus on:
    1. Metrics that have changed significantly from the previous period
    2. Values that are outside the expected range for the industry
    3. Unusual patterns or relationships between different metrics
    
    For each anomaly detected, indicate:
    - The metric name
    - The current and previous values
    - The percentage change
    - The severity (high, medium, low)
    - Whether this change is positive or negative for the business
    - A brief explanation of why this change is notable
    """
    
    # Format the metrics data
    metrics = data.get("metrics", {})
    previous = data.get("previous_metrics", {})
    
    user_prompt = f"""
    Please analyze these dealership metrics and identify significant changes or anomalies:
    
    ## Current Period Metrics
    {json.dumps(metrics, indent=2)}
    
    ## Previous Period Metrics
    {json.dumps(previous, indent=2)}
    
    Focus on metrics that have changed by more than 10% or are significantly outside industry norms.
    Format your response as a list of detected anomalies, with each anomaly containing:
    - metric: The name of the metric
    - current_value: The current value
    - previous_value: The previous value
    - percent_change: The percentage change
    - direction: Whether it increased or decreased
    - severity: High, medium, or low
    - is_concern: Whether this is concerning or positive
    - explanation: A brief explanation of why this is notable
    """
    
    return {
        "system": system_prompt,
        "user": user_prompt
    }


def get_scorecard_prompt(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Get prompts for scorecard generation.
    
    Args:
        data: The data to analyze for scorecard
        
    Returns:
        dict: System and user prompts
    """
    system_prompt = """
    You are an automotive dealership data expert creating a performance scorecard.
    Your task is to evaluate the dealership's key metrics against industry benchmarks
    and assign grades (A, B, C, D, F) based on performance.
    
    Grading scale:
    - A: Exceptional (>20% better than benchmark)
    - B: Above average (5-20% better than benchmark)
    - C: Average (within 5% of benchmark)
    - D: Below average (5-20% worse than benchmark)
    - F: Poor (>20% worse than benchmark)
    
    For each metric evaluated, provide:
    - The metric name
    - The current value
    - The benchmark value
    - The grade
    - The percentage difference from benchmark
    - A brief comment on performance
    """
    
    # Format the metrics data
    metrics = data.get("metrics", {})
    benchmarks = data.get("benchmarks", {})
    
    user_prompt = f"""
    Please evaluate these dealership metrics against industry benchmarks:
    
    ## Current Metrics
    {json.dumps(metrics, indent=2)}
    
    ## Industry Benchmarks
    {json.dumps(benchmarks, indent=2)}
    
    For each metric, assign a grade (A, B, C, D, F) based on how the dealership
    is performing compared to the benchmark. Format your response as a scorecard
    with each metric containing:
    - metric: The name of the metric
    - value: The current value
    - benchmark: The benchmark value
    - grade: The assigned grade (A-F)
    - diff_pct: The percentage difference from benchmark
    - comment: A brief assessment of performance
    """
    
    return {
        "system": system_prompt,
        "user": user_prompt
    }


def get_insight_prompt(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Get prompts for the Insight Generation Engine.
    
    Args:
        data: The data to analyze including metrics, changes, benchmarks
        
    Returns:
        dict: System and user prompts
    """
    system_prompt = """
    You are an automotive dealership data expert creating insights for Watchdog AI.
    Analyze the provided dealership data and generate comprehensive insights focused on:
    
    1. Month-over-month performance changes
    2. Comparison against industry benchmarks
    3. Emerging trends across key metrics
    4. High-priority areas requiring attention
    5. Strategic recommendations for improvement
    
    Your insight analysis should follow this structure:
    
    ## Executive Summary
    A concise overview of the dealership's current performance, highlighting 2-3 critical findings
    
    ## Key Findings
    The most significant metrics, changes, and comparisons that dealership management should be aware of
    
    ## Improvement Opportunities
    Areas where performance is below benchmark or trending negatively
    
    ## Strengths
    Areas where performance exceeds benchmarks or is trending positively
    
    ## Recommendations
    5-7 specific, actionable recommendations prioritized by potential impact
    
    Use clear, concise language with specific numbers and percentages. Format your recommendations
    as bullet points for easy reading. Be specific about which metrics to focus on and what actions 
    to take.
    """
    
    # Prepare metrics summary
    current_metrics = data.get("metrics", {})
    previous_metrics = data.get("previous_metrics", {})
    significant_changes = data.get("significant_changes", [])
    benchmark_comparison = data.get("benchmark_comparison", {})
    trends = data.get("trends", {})
    
    # Format metrics summary
    metrics_summary = "## Current Period Metrics\n"
    
    # Add key metrics
    key_metrics = ["total_leads", "lead_to_sale", "appointment_to_sale", 
                  "total_inventory", "avg_days_in_stock", "aged_inventory_pct",
                  "total_sessions", "srp_to_vdp_ratio"]
    
    for metric in key_metrics:
        if metric in current_metrics:
            value = current_metrics[metric]
            if isinstance(value, float) and metric.endswith("_pct") or "_to_" in metric:
                formatted_value = f"{value * 100:.1f}%"
            else:
                formatted_value = f"{value}"
            
            metrics_summary += f"- {metric}: {formatted_value}\n"
    
    # Add significant changes
    if significant_changes:
        metrics_summary += "\n## Significant Changes\n"
        for change in significant_changes[:5]:  # Show top 5 changes
            metric = change.get("metric", "")
            current = change.get("current_value", 0)
            previous = change.get("previous_value", 0)
            pct_change = change.get("percent_change", 0)
            direction = change.get("direction", "")
            
            metrics_summary += f"- {metric}: {direction} from {previous} to {current} ({abs(pct_change) * 100:.1f}%)\n"
    
    # Add benchmark comparisons
    if benchmark_comparison:
        metrics_summary += "\n## Benchmark Comparisons\n"
        for metric, data in benchmark_comparison.items():
            value = data.get("value", 0)
            benchmark = data.get("benchmark", 0)
            grade = data.get("grade", "")
            
            metrics_summary += f"- {metric}: {value} vs benchmark {benchmark} (Grade: {grade})\n"
    
    # Add trends
    if trends:
        metrics_summary += "\n## Trends\n"
        for metric, data in trends.items():
            direction = data.get("direction", "stable")
            strength = data.get("strength", "unknown")
            
            metrics_summary += f"- {metric}: {direction} ({strength})\n"
    
    # User prompt
    user_prompt = f"""
    Please analyze this dealership data and provide comprehensive insights:
    
    {metrics_summary}
    
    Focus on actionable insights that will help dealership management improve performance.
    Identify specific opportunities for improvement and build on existing strengths.
    """
    
    return {
        "system": system_prompt,
        "user": user_prompt
    }