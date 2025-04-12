"""
Claude client for Watchdog AI.
"""
from anthropic import Anthropic
import os
import time
import json
from typing import Dict, List, Any, Optional, Union
from ..data.template_mapper import TemplateMapper


class ClaudeClient:
    """Client for interacting with Claude."""
    
    def __init__(self):
        """Initialize the Claude client."""
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
        self.max_tokens = 4096
        self.temperature = 0.2
    
    def analyze_dataset(self, df_summary: Dict[str, Any], field_sample: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Analyze a dataset using Claude
        
        Args:
            df_summary: Summary information about the DataFrame
            field_sample: Sample values for each field/column
            
        Returns:
            Dict with Claude's analysis
        """
        # Format the data for Claude
        prompt = self._build_dataset_analysis_prompt(df_summary, field_sample)
        
        # Get Claude's response
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system="You are a specialized AI assistant for dealership data analysis. Your job is to analyze dealership data and provide insights about the data quality, structure, and potential issues.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response
        try:
            analysis_text = response.content[0].text
            
            # Try to extract JSON from the response if it contains JSON
            analysis_json = self._extract_json_from_text(analysis_text)
            
            return {
                "analysis": analysis_text,
                "structured_analysis": analysis_json
            }
        except Exception as e:
            return {
                "error": str(e),
                "analysis": response.content[0].text if response.content else "No response received"
            }
    
    def suggest_mappings(self, column_names: List[str], sample_values: Dict[str, List[Any]]) -> Dict[str, str]:
        """
        Suggest standardized field mappings for unknown columns
        
        Args:
            column_names: List of column names
            sample_values: Sample values for each column
            
        Returns:
            Dict mapping original column names to suggested standard names
        """
        # Format the data for Claude
        prompt = self._build_mapping_suggestion_prompt(column_names, sample_values)
        
        # Get Claude's response
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.1,  # Lower temperature for more consistent mapping
            system="You are a specialized AI assistant for dealership data mapping. Your job is to analyze column names and sample values to suggest standardized field mappings.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response
        try:
            suggestion_text = response.content[0].text
            
            # Extract JSON mapping from the response
            mapping_json = self._extract_json_from_text(suggestion_text)
            
            if isinstance(mapping_json, dict):
                return mapping_json
            else:
                # Fallback to parsing the response line by line
                mappings = {}
                lines = suggestion_text.split("\n")
                for line in lines:
                    if ":" in line and "->" in line:
                        parts = line.split("->")
                        if len(parts) == 2:
                            original = parts[0].split(":")[0].strip()
                            suggested = parts[1].strip()
                            mappings[original] = suggested
                
                return mappings
        except Exception as e:
            return {"error": str(e)}
    
    def answer_question(self, prompt: str, context: dict = None) -> str:
        """
        Get an answer from Claude.
        
        Args:
            prompt: The question or prompt for Claude
            context: Optional context data
            
        Returns:
            str: Claude's response
        """
        # Create the message
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        return message.content[0].text
    
    def _build_dataset_analysis_prompt(self, df_summary: Dict[str, Any], field_sample: Dict[str, List[Any]]) -> str:
        """
        Build prompt for dataset analysis
        
        Args:
            df_summary: Summary information about the DataFrame
            field_sample: Sample values for each field/column
            
        Returns:
            Formatted prompt string
        """
        prompt = """Please analyze this dealership data. I'll provide a summary of the dataset and sample values for each field.

DATASET SUMMARY:
"""
        prompt += json.dumps(df_summary, indent=2)
        
        prompt += "\n\nSAMPLE VALUES FOR EACH FIELD:\n"
        prompt += json.dumps(field_sample, indent=2)
        
        prompt += """\n\nPlease provide a comprehensive analysis with the following sections:

1. Data Overview: Brief description of what kind of dealership data this appears to be
2. Data Quality: Identify any quality issues (missing values, inconsistent formats, outliers)
3. Field Analysis: For each important field, explain its content and potential issues
4. Recommendations: Suggest any data cleaning or transformation steps that should be taken
5. Insights: Identify any interesting patterns or notable characteristics in the data

Please format your response as JSON with these sections as keys. Use markdown formatting within the JSON string values for readability."""
        
        return prompt
    
    def _build_mapping_suggestion_prompt(self, column_names: List[str], sample_values: Dict[str, List[Any]]) -> str:
        """
        Build prompt for mapping suggestion
        
        Args:
            column_names: List of column names
            sample_values: Sample values for each column
            
        Returns:
            Formatted prompt string
        """
        # Get standard field categories from TemplateMapper
        mapper = TemplateMapper()
        standard_categories = {}
        for category, fields in mapper.field_mappings.items():
            standard_categories[category.replace("_fields", "")] = fields[:10]  # Limit to 10 examples
        
        prompt = """I need help mapping these dealership data columns to standardized field names. Below are the column names and sample values.

COLUMN NAMES:
"""
        prompt += json.dumps(column_names, indent=2)
        
        prompt += "\n\nSAMPLE VALUES FOR EACH COLUMN:\n"
        prompt += json.dumps(sample_values, indent=2)
        
        prompt += "\n\nSTANDARD FIELD CATEGORIES:\n"
        prompt += json.dumps(standard_categories, indent=2)
        
        prompt += """\n\nFor each column, please suggest the most appropriate standardized field name based on the column name and sample values. Consider the dealership data context and the standard field categories provided.

Please format your response as a JSON object mapping original column names to suggested standardized field names. For example:
{
  "CUST_EMAIL": "email",
  "VEH_STOCK_NUM": "stock_number",
  "SALE_DT": "sale_date"
}"""
        
        return prompt
    
    def _extract_json_from_text(self, text: str) -> Union[Dict, List, None]:
        """
        Extract JSON from text response
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON as dict/list or None if extraction fails
        """
        # Look for JSON blocks in code fences
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try to find JSON between curly braces
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # If no JSON found or parsing failed, return None
        return None