import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import re
from ..template_mapper import TemplateMapper
from ...utils.exceptions import (
    ValidationError, 
    MissingRequiredFieldError,
    FormatValidationError,
    DataTypeError,
    OutlierError,
    DuplicateDataError,
    MissingValueError
)


class DataValidator:
    """
    Validates and checks data quality in dealership dataframes with enhanced
    error handling and recovery mechanisms.
    """
    
    def __init__(self, 
                mapper: Optional[TemplateMapper] = None, 
                raise_on_error: bool = False,
                log_level: int = logging.INFO):
        """
        Initialize the data validator
        
        Args:
            mapper: Optional TemplateMapper for field categorization
            raise_on_error: Whether to raise exceptions for critical validation errors
            log_level: Logging level (default: INFO)
        """
        self.mapper = mapper or TemplateMapper()
        self.raise_on_error = raise_on_error
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Common data quality checks
        self.checks = {
            "missing_values": self._check_missing_values,
            "duplicate_rows": self._check_duplicate_rows,
            "date_ranges": self._check_date_ranges,
            "numeric_ranges": self._check_numeric_ranges,
            "required_fields": self._check_required_fields,
            "format_consistency": self._check_format_consistency,
            "data_type_consistency": self._check_data_type_consistency,
            "distribution_analysis": self._check_distribution_analysis
        }
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks on a dataframe
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with validation results
            
        Raises:
            ValidationError: If the dataframe fails critical validation checks and raise_on_error is True
            TypeError: If the input is not a pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if df.empty:
            raise ValidationError("Cannot validate empty DataFrame", 
                                 row_count=0, column_count=0)
        
        results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "check_results": {},
            "row_count": len(df),
            "column_count": len(df.columns)
        }
        
        # Run all checks with error handling
        for check_name, check_func in self.checks.items():
            try:
                check_result = check_func(df)
                results["check_results"][check_name] = check_result
                
                # If the check found critical issues
                if not check_result.get("valid", True):
                    results["is_valid"] = False
                    results["issues"].extend(check_result.get("issues", []))
                
                # Add any warnings
                if check_result.get("warnings"):
                    results["warnings"].extend(check_result.get("warnings", []))
                    
            except Exception as e:
                self.logger.error(f"Error in {check_name} validation: {str(e)}", exc_info=True)
                results["check_results"][check_name] = {
                    "valid": False,
                    "error": str(e),
                    "issues": [f"Error in {check_name} validation: {str(e)}"]
                }
                results["is_valid"] = False
                results["issues"].append(f"Error in {check_name} validation: {str(e)}")
        
        # If configured to raise on errors and validation failed
        if self.raise_on_error and not results["is_valid"]:
            raise ValidationError(
                f"Data validation failed with {len(results['issues'])} critical issues",
                context=results
            )
            
        return results
    
    def validate_critical(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run critical validation checks that raise exceptions on failure
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with validation results
            
        Raises:
            Various validation exceptions based on the specific validation failures
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if df.empty:
            raise ValidationError("Cannot validate empty DataFrame", 
                                  row_count=0, column_count=0)
        
        # Check for required fields (will raise MissingRequiredFieldError if missing)
        self._check_required_fields_critical(df)
        
        # Check for excessive missing values (will raise MissingValueError if found)
        self._check_missing_values_critical(df)
        
        # Check for excessive duplicates (will raise DuplicateDataError if found)
        self._check_duplicate_rows_critical(df)
        
        # Run standard validation after critical checks passed
        return self.validate(df)
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for missing values in the dataframe
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with check results
        """
        result = {"valid": True, "issues": [], "warnings": []}
        
        try:
            # Calculate percentage of missing values per column
            missing_pct = (df.isnull().sum() / len(df)) * 100
            cols_with_missing = missing_pct[missing_pct > 0]
            
            if not cols_with_missing.empty:
                result["missing_columns"] = cols_with_missing.to_dict()
                
                # Critical issue if any column has more than 20% missing
                critical_missing = cols_with_missing[cols_with_missing > 20]
                if not critical_missing.empty:
                    result["valid"] = False
                    for col, pct in critical_missing.items():
                        issue = f"Column '{col}' has {pct:.1f}% missing values"
                        result["issues"].append(issue)
                        self.logger.warning(issue)
                
                # Warnings for columns with less than 20% missing
                warning_missing = cols_with_missing[cols_with_missing <= 20]
                if not warning_missing.empty:
                    for col, pct in warning_missing.items():
                        warning = f"Column '{col}' has {pct:.1f}% missing values"
                        result["warnings"].append(warning)
                        self.logger.info(warning)
                        
        except Exception as e:
            self.logger.error(f"Error checking missing values: {str(e)}", exc_info=True)
            result["valid"] = False
            result["issues"].append(f"Error checking missing values: {str(e)}")
        
        return result
    
    def _check_missing_values_critical(self, df: pd.DataFrame) -> None:
        """
        Check for missing values in the dataframe and raise exceptions for critical issues
        
        Args:
            df: DataFrame to check
            
        Raises:
            MissingValueError: If any column has more than 20% missing values
        """
        # Calculate percentage of missing values per column
        missing_pct = (df.isnull().sum() / len(df)) * 100
        cols_with_missing = missing_pct[missing_pct > 0]
        
        # Check each column with missing values
        for col, pct in cols_with_missing.items():
            # Critical issue if any column has more than 20% missing
            if pct > 20:
                missing_count = df[col].isnull().sum()
                self.logger.error(f"Critical: Column '{col}' has {pct:.1f}% missing values")
                
                raise MissingValueError(
                    f"Column '{col}' has excessive missing values ({pct:.1f}%)",
                    field_name=col,
                    missing_count=missing_count,
                    missing_percentage=pct,
                    row_count=len(df)
                )
    
    def _check_duplicate_rows(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for duplicate rows in the dataframe
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with check results
        """
        result = {"valid": True, "issues": [], "warnings": []}
        
        try:
            # Count duplicates
            duplicates = df.duplicated()
            duplicate_count = duplicates.sum()
            
            if duplicate_count > 0:
                duplicate_pct = (duplicate_count / len(df)) * 100
                result["duplicate_count"] = int(duplicate_count)
                result["duplicate_percentage"] = float(duplicate_pct)
                
                # Critical issue if more than 5% duplicates
                if duplicate_pct > 5:
                    result["valid"] = False
                    issue = f"High number of duplicate rows: {duplicate_count} ({duplicate_pct:.1f}%)"
                    result["issues"].append(issue)
                    self.logger.warning(issue)
                else:
                    warning = f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)"
                    result["warnings"].append(warning)
                    self.logger.info(warning)
                    
        except Exception as e:
            self.logger.error(f"Error checking duplicate rows: {str(e)}", exc_info=True)
            result["valid"] = False
            result["issues"].append(f"Error checking duplicate rows: {str(e)}")
        
        return result
    
    def _check_duplicate_rows_critical(self, df: pd.DataFrame) -> None:
        """
        Check for duplicate rows in the dataframe and raise exceptions for critical issues
        
        Args:
            df: DataFrame to check
            
        Raises:
            DuplicateDataError: If more than 5% of rows are duplicates
        """
        # Count duplicates
        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            
            # Critical issue if more than 5% duplicates
            if duplicate_pct > 5:
                # Get indices of duplicate rows for context
                duplicate_indices = df[duplicates].index.tolist()
                duplicate_sample = duplicate_indices[:5]  # Sample of up to 5 indices
                
                self.logger.error(
                    f"Critical: High number of duplicate rows: {duplicate_count} ({duplicate_pct:.1f}%)"
                )
                
                raise DuplicateDataError(
                    f"High number of duplicate rows detected: {duplicate_count} ({duplicate_pct:.1f}%)",
                    duplicate_count=duplicate_count,
                    duplicate_percentage=duplicate_pct,
                    duplicate_indices=duplicate_sample,
                    row_count=len(df)
                )
    
    def _check_date_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check that date columns have reasonable ranges
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with check results
        """
        result = {"valid": True, "issues": [], "warnings": []}
        
        try:
            # Get date columns using mapper
            date_columns = [col for col in df.columns 
                          if self.mapper.get_field_category(col) == "date_fields"]
            
            # Convert columns to datetime if not already
            date_info = {}
            for col in date_columns:
                try:
                    date_series = pd.to_datetime(df[col], errors='coerce')
                    non_null = date_series.dropna()
                    
                    # Skip if no valid dates
                    if len(non_null) == 0:
                        continue
                        
                    # Get min and max dates
                    min_date = non_null.min()
                    max_date = non_null.max()
                    date_info[col] = {
                        "min": min_date,
                        "max": max_date,
                        "range_days": (max_date - min_date).days
                    }
                    
                    # Check for dates in the far future (more than 2 years ahead)
                    future_cutoff = pd.Timestamp.now() + pd.DateOffset(years=2)
                    future_dates = (date_series > future_cutoff).sum()
                    if future_dates > 0:
                        pct_future = (future_dates / len(non_null)) * 100
                        if pct_future > 5:  # Critical if >5% dates are too far in future
                            result["valid"] = False
                            issue = f"Column '{col}' has {future_dates} dates more than 2 years in the future ({pct_future:.1f}%)"
                            result["issues"].append(issue)
                            self.logger.warning(issue)
                        else:
                            warning = f"Column '{col}' has {future_dates} dates more than 2 years in the future"
                            result["warnings"].append(warning)
                            self.logger.info(warning)
                    
                    # Check for dates in the distant past (more than 5 years ago)
                    past_cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
                    past_dates = (date_series < past_cutoff).sum()
                    if past_dates > 0:
                        pct_past = (past_dates / len(non_null)) * 100
                        if pct_past > 10:  # Critical if >10% dates are too far in past
                            result["valid"] = False
                            issue = f"Column '{col}' has {past_dates} dates more than 5 years in the past ({pct_past:.1f}%)"
                            result["issues"].append(issue)
                            self.logger.warning(issue)
                        else:
                            warning = f"Column '{col}' has {past_dates} dates more than 5 years in the past"
                            result["warnings"].append(warning)
                            self.logger.info(warning)
                
                except Exception as e:
                    warning = f"Could not analyze dates in column '{col}': {str(e)}"
                    result["warnings"].append(warning)
                    self.logger.info(warning)
            
            result["date_info"] = date_info
            
        except Exception as e:
            self.logger.error(f"Error checking date ranges: {str(e)}", exc_info=True)
            result["valid"] = False
            result["issues"].append(f"Error checking date ranges: {str(e)}")
        
        return result
    
    def _check_numeric_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check that numeric columns have reasonable values
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with check results
        """
        result = {"valid": True, "issues": [], "warnings": []}
        numeric_info = {}
        
        try:
            # Check numeric columns
            numeric_columns = df.select_dtypes(include=['number']).columns
            
            for col in numeric_columns:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                    
                # Get basic stats
                stats = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                }
                numeric_info[col] = stats
                
                # Check for specific column types using category detection
                col_lower = col.lower()
                
                # Price checks
                if any(term in col_lower for term in ['price', 'cost', 'msrp']):
                    # Check for zero prices
                    zero_prices = (col_data == 0).sum()
                    if zero_prices > 0:
                        pct_zero = (zero_prices / len(col_data)) * 100
                        if pct_zero > 5:  # Critical if >5% are zero
                            result["valid"] = False
                            issue = f"Column '{col}' has {zero_prices} zero values ({pct_zero:.1f}%)"
                            result["issues"].append(issue)
                            self.logger.warning(issue)
                        else:
                            warning = f"Column '{col}' has {zero_prices} zero values"
                            result["warnings"].append(warning)
                            self.logger.info(warning)
                    
                    # Check for very high prices (> $200,000)
                    high_prices = (col_data > 200000).sum()
                    if high_prices > 0:
                        pct_high = (high_prices / len(col_data)) * 100
                        if pct_high > 5:  # Critical if >5% are very high
                            result["valid"] = False
                            issue = f"Column '{col}' has {high_prices} values over $200,000 ({pct_high:.1f}%)"
                            result["issues"].append(issue)
                            self.logger.warning(issue)
                        else:
                            warning = f"Column '{col}' has {high_prices} values over $200,000"
                            result["warnings"].append(warning)
                            self.logger.info(warning)
                
                # Year checks
                if col_lower == 'year' or 'year' in col_lower:
                    current_year = pd.Timestamp.now().year
                    future_years = (col_data > current_year + 2).sum()
                    if future_years > 0:
                        pct_future = (future_years / len(col_data)) * 100
                        if pct_future > 5:  # Critical if >5% are future years
                            result["valid"] = False
                            issue = f"Column '{col}' has {future_years} years more than 2 years in the future ({pct_future:.1f}%)"
                            result["issues"].append(issue)
                            self.logger.warning(issue)
                        else:
                            warning = f"Column '{col}' has {future_years} years more than 2 years in the future"
                            result["warnings"].append(warning)
                            self.logger.info(warning)
                    
                    old_years = (col_data < current_year - 25).sum()
                    if old_years > 0:
                        pct_old = (old_years / len(col_data)) * 100
                        if pct_old > 10:  # Warning if >10% are old years
                            warning = f"Column '{col}' has {old_years} years more than 25 years old ({pct_old:.1f}%)"
                            result["warnings"].append(warning)
                            self.logger.info(warning)
            
            result["numeric_info"] = numeric_info
            
        except Exception as e:
            self.logger.error(f"Error checking numeric ranges: {str(e)}", exc_info=True)
            result["valid"] = False
            result["issues"].append(f"Error checking numeric ranges: {str(e)}")
        
        return result
    
    def _check_required_fields(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check that required fields are present
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with check results
        """
        result = {"valid": True, "issues": [], "warnings": []}
        
        try:
            # Get column categories
            categorized_columns = {}
            for col in df.columns:
                category = self.mapper.get_field_category(col)
                if category:
                    if category not in categorized_columns:
                        categorized_columns[category] = []
                    categorized_columns[category].append(col)
            
            result["categorized_columns"] = categorized_columns
            
            # Required categories (adjust as needed)
            required_categories = [
                "date_fields",
                "lead_source_fields",
                "salesperson_fields"
            ]
            
            missing_categories = []
            for category in required_categories:
                if category not in categorized_columns or not categorized_columns[category]:
                    missing_categories.append(category)
                    result["valid"] = False
                    issue = f"Missing required category: {category.replace('_fields', '')}"
                    result["issues"].append(issue)
                    self.logger.warning(issue)
            
            if missing_categories:
                result["missing_categories"] = missing_categories
                
        except Exception as e:
            self.logger.error(f"Error checking required fields: {str(e)}", exc_info=True)
            result["valid"] = False
            result["issues"].append(f"Error checking required fields: {str(e)}")
        
        return result
    
    def _check_required_fields_critical(self, df: pd.DataFrame) -> None:
        """
        Check that required fields are present and raise exception if not
        
        Args:
            df: DataFrame to check
            
        Raises:
            MissingRequiredFieldError: If any required field category is missing
        """
        # Get column categories
        categorized_columns = {}
        available_fields = set(df.columns)
        
        for col in df.columns:
            category = self.mapper.get_field_category(col)
            if category:
                if category not in categorized_columns:
                    categorized_columns[category] = []
                categorized_columns[category].append(col)
        
        # Required categories (adjust as needed)
        required_categories = [
            "date_fields",
            "lead_source_fields",
            "salesperson_fields"
        ]
        
        # Check for missing categories
        missing_categories = []
        for category in required_categories:
            if category not in categorized_columns or not categorized_columns[category]:
                missing_categories.append(category)
        
        # If any categories are missing, suggest potential matches
        if missing_categories:
            # Generate field suggestions for missing categories
            suggestions = {}
            
            for category in missing_categories:
                # Get potential matches for this category using template mapper
                suggested_fields = self._suggest_fields_for_category(df.columns, category)
                if suggested_fields:
                    suggestions[category] = suggested_fields
            
            # Format for error message
            missing_str = ", ".join([cat.replace("_fields", "") for cat in missing_categories])
            self.logger.error(f"Critical: Missing required categories: {missing_str}")
            
            raise MissingRequiredFieldError(
                f"Missing required data categories: {missing_str}",
                missing_fields=missing_categories,
                suggested_fields=suggestions,
                available_fields=list(available_fields)
            )
    
    def _suggest_fields_for_category(self, columns: List[str], category: str) -> List[str]:
        """
        Suggest potential field matches for a missing category
        
        Args:
            columns: List of column names
            category: Category to find matches for
            
        Returns:
            List of potential field matches for the category
        """
        # Get relevant keywords based on category
        keywords = {
            "date_fields": ["date", "time", "day", "created", "modified", "timestamp"],
            "lead_source_fields": ["source", "lead", "campaign", "referral", "channel", "origin"],
            "salesperson_fields": ["sales", "rep", "agent", "person", "associate", "employee"]
        }
        
        category_keywords = keywords.get(category, [])
        suggestions = []
        
        # Look for columns that might match this category
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in category_keywords):
                suggestions.append(col)
                
        return suggestions
    
    def _check_format_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for format consistency in string columns
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with check results
        """
        result = {"valid": True, "issues": [], "warnings": []}
        format_issues = {}
        
        try:
            # Check string columns
            string_columns = df.select_dtypes(include=['object']).columns
            
            for col in string_columns:
                col_data = df[col].dropna().astype(str)
                if len(col_data) < 10:  # Skip columns with very few values
                    continue
                
                col_lower = col.lower()
                
                # Check email format for email columns
                if 'email' in col_lower:
                    # Improved email regex pattern
                    valid_email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    try:
                        # Use Series.str.match for vectorized operations
                        invalid_emails = ~col_data.str.match(valid_email_pattern)
                        invalid_count = invalid_emails.sum()
                        
                        if invalid_count > 0:
                            invalid_pct = (invalid_count / len(col_data)) * 100
                            format_issues[col] = {
                                "type": "email",
                                "invalid_count": int(invalid_count),
                                "invalid_percentage": float(invalid_pct),
                                "samples": col_data[invalid_emails].head(3).tolist()
                            }
                            
                            if invalid_pct > 20:  # Critical if >20% are invalid
                                result["valid"] = False
                                issue = f"Column '{col}' has {invalid_count} invalid email formats ({invalid_pct:.1f}%)"
                                result["issues"].append(issue)
                                self.logger.warning(issue)
                            else:
                                warning = f"Column '{col}' has {invalid_count} invalid email formats ({invalid_pct:.1f}%)"
                                result["warnings"].append(warning)
                                self.logger.info(warning)
                    except Exception as e:
                        self.logger.error(f"Error checking email format in column '{col}': {str(e)}")
                
                # Check phone format for phone columns
                elif any(term in col_lower for term in ['phone', 'mobile', 'cell']):
                    try:
                        # Remove all non-digit characters and check length
                        clean_phones = col_data.str.replace(r'\D', '', regex=True)
                        invalid_phones = ~clean_phones.str.len().between(10, 15)
                        invalid_count = invalid_phones.sum()
                        
                        if invalid_count > 0:
                            invalid_pct = (invalid_count / len(col_data)) * 100
                            format_issues[col] = {
                                "type": "phone",
                                "invalid_count": int(invalid_count),
                                "invalid_percentage": float(invalid_pct),
                                "samples": col_data[invalid_phones].head(3).tolist()
                            }
                            
                            if invalid_pct > 20:  # Critical if >20% are invalid
                                result["valid"] = False
                                issue = f"Column '{col}' has {invalid_count} invalid phone formats ({invalid_pct:.1f}%)"
                                result["issues"].append(issue)
                                self.logger.warning(issue)
                            else:
                                warning = f"Column '{col}' has {invalid_count} invalid phone formats ({invalid_pct:.1f}%)"
                                result["warnings"].append(warning)
                                self.logger.info(warning)
                    except Exception as e:
                        self.logger.error(f"Error checking phone format in column '{col}': {str(e)}")
                
                # Check VIN format
                elif any(term in col_lower for term in ['vin']):
                    try:
                        # Improved VIN validation - 17 chars, no I,O,Q
                        # VIN must be 17 characters, with a specific character set and no I, O, or Q
                        valid_vin_pattern = r'^[A-HJ-NPR-Z0-9]{17}$'
                        invalid_vins = ~col_data.str.match(valid_vin_pattern)
                        invalid_count = invalid_vins.sum()
                        
                        if invalid_count > 0:
                            invalid_pct = (invalid_count / len(col_data)) * 100
                            format_issues[col] = {
                                "type": "vin",
                                "invalid_count": int(invalid_count),
                                "invalid_percentage": float(invalid_pct),
                                "samples": col_data[invalid_vins].head(3).tolist()
                            }
                            
                            if invalid_pct > 10:  # Critical if >10% are invalid
                                result["valid"] = False
                                issue = f"Column '{col}' has {invalid_count} invalid VIN formats ({invalid_pct:.1f}%)"
                                result["issues"].append(issue)
                                self.logger.warning(issue)
                            else:
                                warning = f"Column '{col}' has {invalid_count} invalid VIN formats ({invalid_pct:.1f}%)"
                                result["warnings"].append(warning)
                                self.logger.info(warning)
                    except Exception as e:
                        self.logger.error(f"Error checking VIN format in column '{col}': {str(e)}")
            
            if format_issues:
                result["format_issues"] = format_issues
                
        except Exception as e:
            self.logger.error(f"Error checking format consistency: {str(e)}", exc_info=True)
            result["valid"] = False
            result["issues"].append(f"Error checking format consistency: {str(e)}")
        
        return result
    
    def _check_data_type_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for consistency of data types within columns with enhanced error handling
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with check results
        """
        result = {"valid": True, "issues": [], "warnings": [], "type_issues": {}}
        
        try:
            # Loop through all columns
            for col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                    
                # Skip columns that are already numeric or datetime
                if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_datetime64_dtype(col_data):
                    continue
                    
                # Focus on object (string) columns that might contain mixed types
                if pd.api.types.is_object_dtype(col_data):
                    type_counts = {}
                    
                    # Check each value's type
                    for val in col_data.iloc[:100].values:  # Sample first 100 values for efficiency
                        val_type = type(val).__name__
                        type_counts[val_type] = type_counts.get(val_type, 0) + 1
                    
                    # If more than one type is present, analyze further
                    if len(type_counts) > 1:
                        total_count = sum(type_counts.values())
                        
                        # Calculate percentages
                        type_percentages = {t: (c / total_count) * 100 for t, c in type_counts.items()}
                        
                        # Analyze specific type mixtures
                        
                        # Check for columns that should be numeric but have string values
                        if 'str' in type_counts and ('int' in type_counts or 'float' in type_counts):
                            # Check if column name suggests it should be numeric
                            col_lower = col.lower()
                            numeric_indicators = ['id', 'num', 'count', 'amount', 'price', 'cost', 
                                                'year', 'age', 'days', 'quantity', 'score']
                            
                            if any(ind in col_lower for ind in numeric_indicators):
                                # Column name suggests numeric but has string values
                                str_pct = type_percentages.get('str', 0)
                                
                                # Get samples of mixed types for context
                                mixed_samples = []
                                for type_name in type_counts.keys():
                                    # Find values of this type
                                    if type_name == 'str':
                                        type_matches = col_data[col_data.apply(lambda x: isinstance(x, str))]
                                    elif type_name == 'int':
                                        type_matches = col_data[col_data.apply(lambda x: isinstance(x, int))]
                                    elif type_name == 'float':
                                        type_matches = col_data[col_data.apply(lambda x: isinstance(x, float))]
                                    else:
                                        continue
                                        
                                    # Add samples (up to 2 per type)
                                    if len(type_matches) > 0:
                                        for val in type_matches.head(2):
                                            mixed_samples.append(val)
                                
                                if str_pct > 10:  # Critical if >10% are strings
                                    result["valid"] = False
                                    issue = f"Column '{col}' appears to be numeric but contains {type_counts.get('str', 0)} " \
                                           f"string values ({str_pct:.1f}%)"
                                    result["issues"].append(issue)
                                    self.logger.warning(issue)
                                else:
                                    warning = f"Column '{col}' appears to be numeric but contains {type_counts.get('str', 0)} " \
                                              f"string values ({str_pct:.1f}%)"
                                    result["warnings"].append(warning)
                                    self.logger.info(warning)
                                    
                                # Store detailed info
                                result["type_issues"][col] = {
                                    "expected_type": "numeric",
                                    "type_counts": type_counts,
                                    "type_percentages": type_percentages,
                                    "samples": mixed_samples
                                }
                                
                                # Check if potential numeric values have format issues
                                if 'str' in type_counts:
                                    # Try to detect if strings could be converted to numbers
                                    str_values = col_data[col_data.apply(lambda x: isinstance(x, str))].head(100)
                                    convertible = str_values.str.replace(r'[$,]', '', regex=True).str.match(r'^-?\d+(\.\d+)?$')
                                    pct_convertible = 100 * convertible.sum() / max(1, len(convertible))
                                    
                                    if pct_convertible > 80:
                                        result["type_issues"][col]["recovery_hint"] = "Most string values appear to be numeric but may have special characters ($ or ,)"
                        
                        # Check for columns that should be dates but have string values
                        elif 'str' in type_counts and any(t in type_counts for t in ['datetime', 'Timestamp']):
                            # Check if column name suggests it should be a date
                            col_lower = col.lower()
                            date_indicators = ['date', 'time', 'day', 'month', 'year', 'created', 'updated', 'timestamp']
                            
                            if any(ind in col_lower for ind in date_indicators):
                                # Column name suggests date but has string values
                                str_pct = type_percentages.get('str', 0)
                                
                                # Get samples for context
                                mixed_samples = []
                                # First, get string samples
                                str_matches = col_data[col_data.apply(lambda x: isinstance(x, str))].head(2)
                                for val in str_matches:
                                    mixed_samples.append(val)
                                    
                                # Then get datetime samples
                                dt_samples = col_data[~col_data.apply(lambda x: isinstance(x, str))].head(2)
                                for val in dt_samples:
                                    mixed_samples.append(val)
                                
                                if str_pct > 10:  # Critical if >10% are strings
                                    result["valid"] = False
                                    issue = f"Column '{col}' appears to be datetime but contains {type_counts.get('str', 0)} " \
                                            f"string values ({str_pct:.1f}%)"
                                    result["issues"].append(issue)
                                    self.logger.warning(issue)
                                else:
                                    warning = f"Column '{col}' appears to be datetime but contains {type_counts.get('str', 0)} " \
                                              f"string values ({str_pct:.1f}%)"
                                    result["warnings"].append(warning)
                                    self.logger.info(warning)
                                
                                # Store detailed info
                                result["type_issues"][col] = {
                                    "expected_type": "datetime",
                                    "type_counts": type_counts,
                                    "type_percentages": type_percentages,
                                    "samples": mixed_samples
                                }
                                
                                # Analyze string dates for potential format issues
                                if 'str' in type_counts:
                                    str_values = col_data[col_data.apply(lambda x: isinstance(x, str))].head(100)
                                    
                                    # Check common date patterns
                                    date_patterns = {
                                        'MM/DD/YYYY': r'\d{1,2}/\d{1,2}/\d{4}',
                                        'YYYY-MM-DD': r'\d{4}-\d{1,2}-\d{1,2}',
                                        'DD-MON-YYYY': r'\d{1,2}-[A-Za-z]{3}-\d{4}',
                                        'Word format': r'[A-Za-z]{3,9} \d{1,2},? \d{4}'
                                    }
                                    
                                    for fmt, pattern in date_patterns.items():
                                        matches = str_values.str.match(pattern).sum()
                                        if matches > 0:
                                            if "format_hint" not in result["type_issues"][col]:
                                                result["type_issues"][col]["format_hint"] = []
                                            pct_match = 100 * matches / len(str_values)
                                            result["type_issues"][col]["format_hint"].append(
                                                f"{fmt} format: {matches} values ({pct_match:.1f}%)"
                                            )
                        
                        # For string columns, check for mixed formats
                        elif 'str' in type_counts and type_counts.get('str', 0) / total_count > 0.8:
                            # For string-dominant columns, check for consistent formats
                            str_values = col_data[col_data.apply(lambda x: isinstance(x, str))].head(100)
                            
                            # Check for mixed number formats (some with commas, some without)
                            col_lower = col.lower()
                            if 'price' in col_lower or 'cost' in col_lower or 'amount' in col_lower:
                                # Patterns for price formats
                                comma_format = str_values.str.contains(r'^\$?\d{1,3}(,\d{3})+(\.\d+)?$').fillna(False).sum()
                                no_comma_format = str_values.str.contains(r'^\$?\d+(\.\d+)?$').fillna(False).sum()
                                
                                if comma_format > 0 and no_comma_format > 0:
                                    warning = f"Column '{col}' has inconsistent number formats: {comma_format} with commas, " \
                                            f"{no_comma_format} without commas"
                                    result["warnings"].append(warning)
                                    self.logger.info(warning)
                                    
                                    # Get samples of each format
                                    comma_samples = str_values[str_values.str.contains(r'^\$?\d{1,3}(,\d{3})+(\.\d+)?$', na=False)].head(2).tolist()
                                    no_comma_samples = str_values[str_values.str.contains(r'^\$?\d+(\.\d+)?$', na=False)].head(2).tolist()
                                    format_samples = comma_samples + no_comma_samples
                                    
                                    # Store detailed info
                                    result["type_issues"][col] = {
                                        "issue_type": "inconsistent_format",
                                        "details": {
                                            "with_commas": int(comma_format),
                                            "without_commas": int(no_comma_format)
                                        },
                                        "samples": format_samples
                                    }
                                    
                            # Check for mixed date formats
                            elif any(ind in col_lower for ind in ['date', 'time', 'day', 'month', 'year', 'created', 'updated', 'timestamp']):
                                date_formats = {}
                                format_patterns = {
                                    'MM/DD/YYYY': r'\d{1,2}/\d{1,2}/\d{4}',
                                    'YYYY-MM-DD': r'\d{4}-\d{1,2}-\d{1,2}',
                                    'MM-DD-YYYY': r'\d{1,2}-\d{1,2}-\d{4}',
                                    'other_format': r'\d{1,2}[^0-9]\d{1,2}[^0-9]\d{2,4}'
                                }
                                
                                for fmt, pattern in format_patterns.items():
                                    format_count = str_values.str.match(pattern).fillna(False).sum()
                                    if format_count > 0:
                                        date_formats[fmt] = int(format_count)
                                
                                if len(date_formats) > 1:
                                    warning = f"Column '{col}' has {len(date_formats)} different date formats"
                                    result["warnings"].append(warning)
                                    self.logger.info(warning)
                                    
                                    # Get samples of each format
                                    format_samples = []
                                    for fmt, pattern in format_patterns.items():
                                        if fmt in date_formats:
                                            matches = str_values[str_values.str.match(pattern, na=False)]
                                            if not matches.empty:
                                                for val in matches.head(2):
                                                    format_samples.append(val)
                                    
                                    # Store detailed info
                                    result["type_issues"][col] = {
                                        "issue_type": "inconsistent_date_format",
                                        "format_counts": date_formats,
                                        "samples": format_samples
                                    }
                        
                        # For general mixed types not covered above
                        else:
                            type_list = ", ".join([f"{t}: {c} ({type_percentages[t]:.1f}%)" for t, c in type_counts.items()])
                            warning = f"Column '{col}' contains mixed data types: {type_list}"
                            result["warnings"].append(warning)
                            self.logger.info(warning)
                            
                            # Get samples of different types
                            mixed_samples = []
                            for type_name in type_counts.keys()[:3]:  # Limit to first 3 types
                                # Try to find values of this type
                                if len(mixed_samples) < 5:  # Limit to 5 total samples
                                    try:
                                        if type_name == 'str':
                                            vals = col_data[col_data.apply(lambda x: isinstance(x, str))].head(2)
                                        elif type_name == 'int':
                                            vals = col_data[col_data.apply(lambda x: isinstance(x, int))].head(2)
                                        elif type_name == 'float':
                                            vals = col_data[col_data.apply(lambda x: isinstance(x, float))].head(2)
                                        elif type_name == 'bool':
                                            vals = col_data[col_data.apply(lambda x: isinstance(x, bool))].head(2)
                                        else:
                                            vals = pd.Series([])
                                        
                                        for val in vals:
                                            mixed_samples.append(val)
                                    except Exception:
                                        pass  # Skip if we can't extract samples
                            
                            # Store detailed info
                            result["type_issues"][col] = {
                                "issue_type": "mixed_types",
                                "type_counts": type_counts,
                                "type_percentages": type_percentages,
                                "samples": mixed_samples
                            }
            
        except Exception as e:
            self.logger.error(f"Error checking data type consistency: {str(e)}", exc_info=True)
            result["valid"] = False
            result["issues"].append(f"Error checking data type consistency: {str(e)}")
        
        return result
    
    def _check_distribution_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data distributions to identify outliers and skewed distributions
        with enhanced error handling
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with check results
        """
        result = {"valid": True, "issues": [], "warnings": [], "distribution_details": {}}
        
        try:
            # Only analyze numeric columns
            numeric_columns = df.select_dtypes(include=['number']).columns
            
            if len(numeric_columns) == 0:
                return result
                
            # Check each numeric column
            for col in numeric_columns:
                col_data = df[col].dropna()
                if len(col_data) < 10:  # Skip columns with too few values
                    continue
                    
                try:
                    # Calculate distribution statistics
                    dist_stats = {
                        "count": len(col_data),
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "25%": float(col_data.quantile(0.25)),
                        "75%": float(col_data.quantile(0.75)),
                        "skew": float(col_data.skew()),
                        "kurtosis": float(col_data.kurtosis())
                    }
                    
                    # Calculate IQR for outlier detection
                    q1 = dist_stats["25%"]
                    q3 = dist_stats["75%"]
                    iqr = q3 - q1
                    
                    # Calculate outlier bounds
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)
                    
                    # Find outliers using IQR method
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_percentage = (outlier_count / len(col_data)) * 100
                    
                    # Calculate Z-scores for another outlier detection method
                    z_scores = None
                    if dist_stats["std"] > 0:
                        z_scores = (col_data - dist_stats["mean"]) / dist_stats["std"]
                        extreme_outliers = col_data[abs(z_scores) > 3]  # 3 standard deviations
                        extreme_count = len(extreme_outliers)
                        extreme_percentage = (extreme_count / len(col_data)) * 100
                    else:
                        # If standard deviation is 0, no outliers by z-score method
                        extreme_count = 0
                        extreme_percentage = 0
                    
                    # Store distribution details
                    dist_stats.update({
                        "iqr": float(iqr),
                        "outlier_count": int(outlier_count),
                        "outlier_percentage": float(outlier_percentage),
                        "extreme_outlier_count": int(extreme_count),
                        "extreme_percentage": float(extreme_percentage),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    })
                    
                    # Get column category to help with analysis
                    col_category = self.mapper.get_field_category(col)
                    dist_stats["category"] = col_category
                    
                    # Store sample outliers (up to 5)
                    if outlier_count > 0:
                        dist_stats["outlier_samples"] = outliers.sample(min(5, outlier_count)).tolist()
                    
                    # Check for severe skewness
                    skew_value = dist_stats["skew"]
                    high_skew = abs(skew_value) > 2
                    
                    # Check for bimodal or multimodal distribution
                    kurtosis_value = dist_stats["kurtosis"]
                    potential_multimodal = kurtosis_value < -1
                    
                    # Check for high outlier percentage
                    high_outlier_pct = outlier_percentage > 10
                    high_extreme_pct = extreme_percentage > 5
                    
                    # Analyze specific issues based on column category/name
                    col_lower = col.lower()
                    dist_stats["possible_issues"] = []
                        
                    # Store all distribution stats
                    result["distribution_details"][col] = dist_stats
                    
                    # Flag issues for different column types
                    if "price" in col_lower or "cost" in col_lower or "amount" in col_lower:
                        # Price/cost columns with high outliers
                        if high_outlier_pct:
                            warning = f"Column '{col}' has {outlier_count} price outliers ({outlier_percentage:.1f}%)"
                            result["warnings"].append(warning)
                            dist_stats["possible_issues"].append("high_price_variance")
                            self.logger.info(warning)
                            
                        # Extreme price outliers
                        if high_extreme_pct:
                            issue = f"Column '{col}' has {extreme_count} extreme price outliers ({extreme_percentage:.1f}%)"
                            result["issues"].append(issue)
                            result["valid"] = False
                            dist_stats["possible_issues"].append("extreme_price_outliers")
                            self.logger.warning(issue)
                            
                            # Potentially raise an OutlierError if in critical validation
                            if self.raise_on_error:
                                outlier_samples = outliers.head(5).tolist() if not outliers.empty else []
                                error_context = {
                                    "outlier_count": outlier_count,
                                    "outlier_percentage": outlier_percentage,
                                    "outlier_samples": outlier_samples,
                                    "field_stats": {
                                        "mean": dist_stats["mean"],
                                        "median": dist_stats["median"],
                                        "std": dist_stats["std"]
                                    }
                                }
                                self.logger.error(f"Critical outliers in '{col}': {extreme_count} values ({extreme_percentage:.1f}%)")
                            
                        # Highly skewed price distribution
                        if high_skew:
                            warning = f"Column '{col}' has a highly skewed price distribution (skew={skew_value:.2f})"
                            result["warnings"].append(warning)
                            dist_stats["possible_issues"].append("skewed_price_distribution")
                            self.logger.info(warning)
                            
                    elif "days" in col_lower or "age" in col_lower or "time" in col_lower:
                        # Days/age columns with high outliers
                        if high_outlier_pct:
                            warning = f"Column '{col}' has {outlier_count} time/age outliers ({outlier_percentage:.1f}%)"
                            result["warnings"].append(warning)
                            dist_stats["possible_issues"].append("high_time_variance")
                            self.logger.info(warning)
                            
                        # Extremely skewed time distribution
                        if high_skew and skew_value > 0:  # Positive skew for time often indicates old outliers
                            if extreme_percentage > 2:  # If also have extreme outliers
                                issue = f"Column '{col}' has a highly skewed time distribution (skew={skew_value:.2f})"
                                result["issues"].append(issue)
                                result["valid"] = False
                                self.logger.warning(issue)
                            else:
                                warning = f"Column '{col}' has a highly skewed time distribution (skew={skew_value:.2f})"
                                result["warnings"].append(warning)
                                self.logger.info(warning)
                            dist_stats["possible_issues"].append("skewed_time_distribution")
                            
                    elif "count" in col_lower or "quantity" in col_lower or "number" in col_lower:
                        # Count columns with high outliers
                        if high_outlier_pct:
                            warning = f"Column '{col}' has {outlier_count} count/quantity outliers ({outlier_percentage:.1f}%)"
                            result["warnings"].append(warning)
                            dist_stats["possible_issues"].append("unusual_counts")
                            self.logger.info(warning)
                            
                        # Potential multimodal distribution in counts
                        if potential_multimodal:
                            warning = f"Column '{col}' may have a multimodal distribution (kurtosis={kurtosis_value:.2f})"
                            result["warnings"].append(warning)
                            dist_stats["possible_issues"].append("multimodal_counts")
                            self.logger.info(warning)
                            
                    elif "ratio" in col_lower or "rate" in col_lower or "percentage" in col_lower:
                        # Verify values are within expected range for ratios (0-1 or 0-100)
                        if dist_stats["max"] > 100 and "percentage" in col_lower:
                            issue = f"Column '{col}' has percentage values > 100 (max={dist_stats['max']:.2f})"
                            result["issues"].append(issue)
                            result["valid"] = False
                            dist_stats["possible_issues"].append("invalid_percentage_range")
                            self.logger.warning(issue)
                            
                        elif dist_stats["max"] > 1 and "ratio" in col_lower:
                            # Check if values might be percentages stored as ratios
                            if dist_stats["max"] <= 100:
                                warning = f"Column '{col}' appears to have percentage values but is named as a ratio (max={dist_stats['max']:.2f})"
                                result["warnings"].append(warning)
                                dist_stats["possible_issues"].append("percentage_as_ratio")
                                self.logger.info(warning)
                            else:
                                issue = f"Column '{col}' has ratio values > 1 (max={dist_stats['max']:.2f})"
                                result["issues"].append(issue)
                                result["valid"] = False
                                dist_stats["possible_issues"].append("invalid_ratio_range")
                                self.logger.warning(issue)
                    else:
                        # General numeric columns
                        if high_extreme_pct:
                            warning = f"Column '{col}' has {extreme_count} extreme outliers ({extreme_percentage:.1f}%)"
                            result["warnings"].append(warning)
                            dist_stats["possible_issues"].append("high_outlier_percentage")
                            self.logger.info(warning)
                        
                        # Flag columns with very high skew or unusual distributions
                        if abs(skew_value) > 3:
                            skew_direction = "positive" if skew_value > 0 else "negative"
                            warning = f"Column '{col}' has a very {skew_direction} skewed distribution (skew={skew_value:.2f})"
                            result["warnings"].append(warning)
                            dist_stats["possible_issues"].append(f"{skew_direction}_skew")
                            self.logger.info(warning)
                            
                        # Flag potential multimodal distributions
                        if potential_multimodal:
                            warning = f"Column '{col}' may have a multimodal distribution (kurtosis={kurtosis_value:.2f})"
                            result["warnings"].append(warning)
                            dist_stats["possible_issues"].append("potential_multimodal")
                            self.logger.info(warning)
                            
                except Exception as e:
                    self.logger.error(f"Error analyzing distribution for column '{col}': {str(e)}")
                    warning = f"Could not analyze distribution for column '{col}': {str(e)}"
                    result["warnings"].append(warning)
            
        except Exception as e:
            self.logger.error(f"Error in distribution analysis: {str(e)}", exc_info=True)
            result["valid"] = False
            result["issues"].append(f"Error in distribution analysis: {str(e)}")
        
        return result
    
    def summary_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary report from validation results
        
        Args:
            validation_results: Results from validate() method
            
        Returns:
            Formatted string with validation summary
        """
        try:
            lines = ["# Data Validation Summary"]
            
            # Overall status
            is_valid = validation_results.get("is_valid", False)
            status = "✅ PASSED" if is_valid else "❌ FAILED"
            lines.append(f"\n## Overall Status: {status}\n")
            
            # Add row and column counts if available
            if "row_count" in validation_results and "column_count" in validation_results:
                lines.append(f"Data shape: {validation_results['row_count']} rows × {validation_results['column_count']} columns\n")
            
            # Issues
            issues = validation_results.get("issues", [])
            if issues:
                lines.append("## Critical Issues:")
                for issue in issues:
                    lines.append(f"- {issue}")
                lines.append("")
            
            # Warnings
            warnings = validation_results.get("warnings", [])
            if warnings:
                lines.append("## Warnings:")
                for warning in warnings:
                    lines.append(f"- {warning}")
                lines.append("")
            
            # Add details for each check
            check_results = validation_results.get("check_results", {})
            if check_results:
                lines.append("## Check Details:")
                
                # Missing values
                if "missing_values" in check_results:
                    mv_check = check_results["missing_values"]
                    missing_cols = mv_check.get("missing_columns", {})
                    if missing_cols:
                        lines.append("\n### Missing Values:")
                        for col, pct in missing_cols.items():
                            lines.append(f"- {col}: {pct:.1f}%")
                
                # Duplicates
                if "duplicate_rows" in check_results:
                    dup_check = check_results["duplicate_rows"]
                    dup_count = dup_check.get("duplicate_count", 0)
                    if dup_count > 0:
                        dup_pct = dup_check.get("duplicate_percentage", 0)
                        lines.append(f"\n### Duplicates: {dup_count} rows ({dup_pct:.1f}%)")
                
                # Date ranges
                if "date_ranges" in check_results:
                    date_check = check_results["date_ranges"]
                    date_info = date_check.get("date_info", {})
                    if date_info:
                        lines.append("\n### Date Ranges:")
                        for col, info in date_info.items():
                            min_date = info.get("min").strftime('%Y-%m-%d')
                            max_date = info.get("max").strftime('%Y-%m-%d')
                            range_days = info.get("range_days")
                            lines.append(f"- {col}: {min_date} to {max_date} ({range_days} days)")
                
                # Numeric ranges
                if "numeric_ranges" in check_results:
                    num_check = check_results["numeric_ranges"]
                    num_info = num_check.get("numeric_info", {})
                    if num_info:
                        lines.append("\n### Numeric Ranges:")
                        for col, stats in num_info.items():
                            lines.append(f"- {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
                
                # Format issues
                if "format_consistency" in check_results:
                    format_check = check_results["format_consistency"]
                    format_issues = format_check.get("format_issues", {})
                    if format_issues:
                        lines.append("\n### Format Issues:")
                        for col, issues in format_issues.items():
                            lines.append(
                                f"- {col} ({issues['type']}): {issues['invalid_count']} invalid ({issues['invalid_percentage']:.1f}%)"
                            )
                            # Include sample invalid values if available
                            if 'samples' in issues and issues['samples']:
                                samples_str = ', '.join([str(s) for s in issues['samples'][:3]])
                                lines.append(f"  Sample values: {samples_str}")
                
                # Data type consistency issues
                if "data_type_consistency" in check_results:
                    type_check = check_results["data_type_consistency"]
                    type_issues = type_check.get("type_issues", {})
                    if type_issues:
                        lines.append("\n### Data Type Consistency Issues:")
                        for col, issues in type_issues.items():
                            issue_type = issues.get("issue_type", issues.get("expected_type", "mixed"))
                            
                            # Format different issue types appropriately
                            if issue_type == "numeric":
                                lines.append(f"- {col}: Mixed numeric and string values")
                                type_counts = issues.get("type_counts", {})
                                if type_counts:
                                    counts_str = ", ".join([f"{t}: {c}" for t, c in type_counts.items()])
                                    lines.append(f"  Types: {counts_str}")
                                    
                            elif issue_type == "datetime":
                                lines.append(f"- {col}: Mixed datetime and string values")
                                type_counts = issues.get("type_counts", {})
                                if type_counts:
                                    counts_str = ", ".join([f"{t}: {c}" for t, c in type_counts.items()])
                                    lines.append(f"  Types: {counts_str}")
                                    
                            elif issue_type == "inconsistent_format":
                                details = issues.get("details", {})
                                lines.append(f"- {col}: Inconsistent number formats")
                                lines.append(f"  With commas: {details.get('with_commas', 0)}, Without commas: {details.get('without_commas', 0)}")
                                
                            elif issue_type == "inconsistent_date_format":
                                format_counts = issues.get("format_counts", {})
                                lines.append(f"- {col}: Inconsistent date formats")
                                for fmt, count in format_counts.items():
                                    lines.append(f"  {fmt}: {count}")
                                    
                            elif issue_type == "mixed_types":
                                type_counts = issues.get("type_counts", {})
                                lines.append(f"- {col}: Contains mixed data types")
                                if type_counts:
                                    counts_str = ", ".join([f"{t}: {c}" for t, c in type_counts.items()])
                                    lines.append(f"  Types: {counts_str}")
                            
                            # Show samples for all issue types if available
                            samples = issues.get("samples", [])
                            if samples:
                                sample_str = ", ".join([str(s) for s in samples[:3]])
                                lines.append(f"  Sample values: {sample_str}")
                                
                            # Include recovery hints if available
                            if "recovery_hint" in issues:
                                lines.append(f"  Hint: {issues['recovery_hint']}")
                            if "format_hint" in issues:
                                for hint in issues["format_hint"]:
                                    lines.append(f"  {hint}")
            
            # Add distribution analysis information
            if "distribution_analysis" in check_results:
                dist_check = check_results["distribution_analysis"]
                dist_details = dist_check.get("distribution_details", {})
                if dist_details:
                    lines.append("\n### Distribution Analysis:")
                    for col, stats in dist_details.items():
                        # Only show columns with issues
                        if "possible_issues" in stats and stats["possible_issues"]:
                            lines.append(f"- {col}:")
                            
                            # Show basic stats
                            lines.append(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
                            
                            # Show outlier information
                            outlier_count = stats.get("outlier_count", 0)
                            if outlier_count > 0:
                                lines.append(f"  Outliers: {outlier_count} ({stats['outlier_percentage']:.1f}%)")
                                
                                # List sample outliers if available
                                if "outlier_samples" in stats:
                                    samples = stats["outlier_samples"]
                                    sample_str = ", ".join([str(s) for s in samples[:3]])
                                    lines.append(f"  Sample outliers: {sample_str}")
                            
                            # Show distribution characteristics
                            if abs(stats.get("skew", 0)) > 1:
                                skew_type = "positive" if stats["skew"] > 0 else "negative"
                                lines.append(f"  Skew: {stats['skew']:.2f} ({skew_type})")
                            
                            # List possible issues
                            issues = stats.get("possible_issues", [])
                            if issues:
                                issues_map = {
                                    "high_price_variance": "High variance in prices",
                                    "extreme_price_outliers": "Extreme price outliers detected",
                                    "skewed_price_distribution": "Prices are not normally distributed",
                                    "high_time_variance": "High variance in time/age values",
                                    "skewed_time_distribution": "Time values are skewed (possible old entries)",
                                    "unusual_counts": "Unusual count patterns detected",
                                    "multimodal_counts": "Multiple peaks in count distribution",
                                    "invalid_percentage_range": "Percentage values out of range (>100%)",
                                    "percentage_as_ratio": "Values might be percentages in ratio format",
                                    "invalid_ratio_range": "Ratio values out of range (>1)",
                                    "high_outlier_percentage": "High percentage of outliers",
                                    "positive_skew": "Distribution is positively skewed",
                                    "negative_skew": "Distribution is negatively skewed",
                                    "potential_multimodal": "Distribution may have multiple peaks"
                                }
                                
                                readable_issues = [issues_map.get(issue, issue) for issue in issues]
                                issues_str = ", ".join(readable_issues)
                                lines.append(f"  Possible issues: {issues_str}")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}", exc_info=True)
            return f"Error generating summary report: {str(e)}"