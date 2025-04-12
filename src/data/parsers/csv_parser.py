import pandas as pd
import csv
import os
from typing import Dict, Any, Optional, List, Tuple
from ..template_mapper import TemplateMapper


class CSVParser:
    """Parser for CSV and Excel files"""
    
    def __init__(self, mapper: Optional[TemplateMapper] = None):
        """
        Initialize the CSV parser
        
        Args:
            mapper: Optional TemplateMapper instance for field mapping
        """
        self.mapper = mapper or TemplateMapper()
    
    def parse_file(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse a CSV or Excel file into a pandas DataFrame
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments to pass to pandas read_csv or read_excel
            
        Returns:
            Tuple of (processed_dataframe, processing_results)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Load the file into a DataFrame
        if file_ext in [".csv", ".txt"]:
            df = pd.read_csv(file_path, **kwargs)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        # Process the DataFrame
        processed_df, results = self.mapper.process_dataframe(df)
        results["file_name"] = os.path.basename(file_path)
        results["file_type"] = file_ext
        
        return processed_df, results
    
    def detect_delimiter(self, file_path: str, sample_size: int = 1024) -> str:
        """
        Detect the delimiter used in a CSV file
        
        Args:
            file_path: Path to the CSV file
            sample_size: Number of bytes to sample for detection
            
        Returns:
            Detected delimiter character
        """
        with open(file_path, 'r', newline='') as f:
            sample = f.read(sample_size)
            
        # Count potential delimiters
        delimiters = {
            ',': sample.count(','),
            '\t': sample.count('\t'),
            ';': sample.count(';'),
            '|': sample.count('|')
        }
        
        # Return the most common delimiter
        return max(delimiters.items(), key=lambda x: x[1])[0]
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about a CSV/Excel file without fully loading it
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with file information
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        file_info = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_type": file_ext,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        }
        
        # Get column headers
        if file_ext in [".csv", ".txt"]:
            delimiter = self.detect_delimiter(file_path)
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader, [])
                file_info["columns"] = headers
                file_info["delimiter"] = delimiter
                
                # Count rows (limited sample)
                rows = 0
                for _ in reader:
                    rows += 1
                    if rows >= 1000:  # Limit row counting for large files
                        rows = f"{rows}+"
                        break
                file_info["row_count"] = rows
        
        elif file_ext in [".xlsx", ".xls"]:
            # Excel files - get sheet names and first row as headers
            xls = pd.ExcelFile(file_path)
            file_info["sheets"] = xls.sheet_names
            # Get headers from the first sheet
            if xls.sheet_names:
                df_sample = pd.read_excel(file_path, sheet_name=xls.sheet_names[0], nrows=1)
                file_info["columns"] = list(df_sample.columns)
        
        return file_info
    
    def batch_process(self, file_paths: List[str], **kwargs) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Process multiple files in batch mode
        
        Args:
            file_paths: List of file paths to process
            **kwargs: Additional arguments to pass to pandas read functions
            
        Returns:
            Dict mapping file names to (processed_dataframe, results) tuples
        """
        results = {}
        for file_path in file_paths:
            try:
                df, info = self.parse_file(file_path, **kwargs)
                results[os.path.basename(file_path)] = (df, info)
            except Exception as e:
                results[os.path.basename(file_path)] = (None, {"error": str(e)})
                
        return results