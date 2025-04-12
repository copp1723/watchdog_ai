import pandas as pd
import json
import re
import os
from typing import Dict, List, Optional, Tuple, Union, Any


class TemplateMapper:
    """Maps dealership data fields to standardized formats."""
    
    def __init__(self, dictionary_path: str = "assets/data_dictionary.json"):
        """
        Initialize the mapper with a data dictionary.
        
        Args:
            dictionary_path: Path to the data dictionary JSON file
        """
        # Load the data dictionary
        if os.path.exists(dictionary_path):
            with open(dictionary_path, 'r') as f:
                self.dictionary = json.load(f)
        else:
            raise FileNotFoundError(f"Dictionary file not found at {dictionary_path}")
        
        # Extract field mappings for easier access
        self.field_mappings = self.dictionary.get("field_mappings", {})
        self.doc_templates = self.dictionary.get("document_templates", {})
        
        # Create inverse mapping for quick lookups
        self.field_to_category = {}
        for category, fields in self.field_mappings.items():
            for field in fields:
                self.field_to_category[field] = category
    
    def detect_template(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect which predefined template best matches the dataframe columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            tuple: (template_name, confidence_score)
        """
        best_match = (None, 0.0)
        columns = [col.lower() for col in df.columns]
        
        for template_name, template_info in self.doc_templates.items():
            key_fields = [field.lower() for field in template_info.get("key_fields", [])]
            if not key_fields:
                continue
            
            # Calculate match percentage based on key fields
            matches = sum(1 for field in key_fields if any(
                self._is_similar_field(field, col) for col in columns
            ))
            
            score = matches / len(key_fields)
            
            if score > best_match[1]:
                best_match = (template_name, score)
        
        return best_match
    
    def _is_similar_field(self, field1: str, field2: str) -> bool:
        """Check if two field names are similar (ignoring spaces, underscores, case)."""
        field1 = self._normalize_field_name(field1)
        field2 = self._normalize_field_name(field2)
        return field1 == field2
    
    def _normalize_field_name(self, field: str) -> str:
        """Normalize field name by removing spaces, underscores, and lowercasing."""
        return re.sub(r'[^a-z0-9]', '', field.lower())
    
    def get_field_category(self, field: str) -> Optional[str]:
        """
        Determine the category of a field (date, lead_source, etc.)
        
        Args:
            field: Field name to categorize
            
        Returns:
            str or None: Category name or None if not found
        """
        normalized = self._normalize_field_name(field)
        
        # First check exact matches
        for category, fields in self.field_mappings.items():
            if any(self._normalize_field_name(f) == normalized for f in fields):
                return category
        
        # Then check partial matches
        for category, fields in self.field_mappings.items():
            if any(self._normalize_field_name(f) in normalized or 
                   normalized in self._normalize_field_name(f) for f in fields):
                return category
                
        return None
    
    def get_standardized_name(self, field: str) -> str:
        """
        Get a standardized field name based on the category.
        
        Args:
            field: Original field name
            
        Returns:
            str: Standardized field name
        """
        category = self.get_field_category(field)
        if not category:
            return field  # Keep original if no category found
        
        # Map to standard names based on category
        category_to_standard = {
            "date_fields": "date",
            "lead_source_fields": "lead_source",
            "salesperson_fields": "salesperson",
            "inventory_fields": {
                "stock": "stock_number",
                "vin": "vin",
                "days": "days_in_stock",
                "age": "days_in_stock"
            },
            "vehicle_fields": lambda f: self._normalize_field_name(f).replace("ext", "exterior").replace("int", "interior")
        }
        
        if category in category_to_standard:
            if isinstance(category_to_standard[category], dict):
                # For categories with multiple standard names
                field_norm = self._normalize_field_name(field)
                for key, value in category_to_standard[category].items():
                    if key in field_norm:
                        return value
                # Fallback to first match in the category
                return self.field_mappings[category][0]
            elif callable(category_to_standard[category]):
                # For categories with custom mapping logic
                return category_to_standard[category](field)
            else:
                # For categories with a single standard name
                return category_to_standard[category]
        
        return field
    
    def map_dataframe(self, df: pd.DataFrame, template_hint: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Map a dataframe's columns to standardized names.
        
        Args:
            df: DataFrame to map
            template_hint: Optional hint about which template to use
            
        Returns:
            tuple: (mapped_dataframe, mapping_dict)
        """
        # Detect template if not provided
        if not template_hint:
            template_name, confidence = self.detect_template(df)
            template_hint = template_name if confidence > 0.6 else None
        
        # Create column mapping
        mapping = {}
        for col in df.columns:
            std_name = self.get_standardized_name(col)
            if std_name != col:
                mapping[col] = std_name
        
        # Apply mapping
        if mapping:
            df = df.rename(columns=mapping)
        
        return df, mapping
    
    def normalize_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date columns to pandas datetime format.
        
        Args:
            df: DataFrame with columns to normalize
            
        Returns:
            DataFrame: DataFrame with normalized date columns
        """
        date_columns = [col for col in df.columns 
                      if self.get_field_category(col) == "date_fields"]
        
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                print(f"Could not convert {col} to datetime: {e}")
                
        return df
    
    def normalize_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to convert appropriate columns to numeric format.
        
        Args:
            df: DataFrame with columns to normalize
            
        Returns:
            DataFrame: DataFrame with normalized numeric columns
        """
        # Numeric indicators in column names
        numeric_indicators = [
            'price', 'cost', 'amount', 'count', 'number', 'total', 'sum',
            'days', 'age', 'year', 'mileage', 'miles', 'odometer'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            # Skip date columns
            if self.get_field_category(col) == "date_fields":
                continue
                
            # Check if column likely contains numeric data
            if any(ind in col_lower for ind in numeric_indicators):
                try:
                    # Try to convert to numeric, coercing errors
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"Could not convert {col} to numeric: {e}")
                    
        return df
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a dataframe for common issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            dict: Validation results with issues found
        """
        result = {
            "is_valid": True,
            "issues": [],
            "null_counts": {},
            "column_categories": {}
        }
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            result["null_counts"] = null_counts[null_counts > 0].to_dict()
            if (null_counts / len(df) > 0.2).any():
                result["is_valid"] = False
                result["issues"].append("High percentage of missing values")
        
        # Categorize columns
        for col in df.columns:
            category = self.get_field_category(col)
            if category:
                if category not in result["column_categories"]:
                    result["column_categories"][category] = []
                result["column_categories"][category].append(col)
        
        # Check for required categories
        required_categories = ["date_fields"]
        for category in required_categories:
            if category not in result["column_categories"]:
                result["is_valid"] = False
                result["issues"].append(f"Missing {category.replace('_fields', '')} columns")
        
        return result
    
    def process_dataframe(self, df: pd.DataFrame, template_hint: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete processing of a dataframe - mapping, normalization, and validation.
        
        Args:
            df: DataFrame to process
            template_hint: Optional hint about which template to use
            
        Returns:
            tuple: (processed_dataframe, processing_results)
        """
        results = {
            "original_columns": list(df.columns),
            "row_count": len(df)
        }
        
        # Detect template
        template_name, confidence = self.detect_template(df)
        results["detected_template"] = template_name
        results["template_confidence"] = confidence
        
        # Map columns
        df, mapping = self.map_dataframe(df, template_hint or template_name)
        results["column_mapping"] = mapping
        
        # Normalize dates and numbers
        df = self.normalize_date_columns(df)
        df = self.normalize_numeric_columns(df)
        
        # Validate
        validation = self.validate_dataframe(df)
        results.update(validation)
        
        return df, results