import camelot
import pandas as pd
import os
import io
from typing import List, Dict, Any, Tuple, Optional
from pdfminer.high_level import extract_text
import re
from ..template_mapper import TemplateMapper


class PDFParser:
    """Parser for PDF files with structured data (tables)"""
    
    def __init__(self, mapper: Optional[TemplateMapper] = None):
        """
        Initialize the PDF parser
        
        Args:
            mapper: Optional TemplateMapper instance for field mapping
        """
        self.mapper = mapper or TemplateMapper()
        self.extraction_methods = ["lattice", "stream"]
    
    def parse_file(self, file_path: str, pages: str = "all") -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
        """
        Parse tables from a PDF file
        
        Args:
            file_path: Path to the PDF file
            pages: Pages to extract tables from (e.g., "1,3,4-10" or "all")
            
        Returns:
            Tuple of (list of DataFrames, extraction results)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        results = {
            "file_name": os.path.basename(file_path),
            "tables_found": 0,
            "pages_processed": 0,
            "extraction_method": None,
            "extraction_score": 0,
            "processing_issues": []
        }
        
        tables = []
        
        # Try different extraction methods
        for method in self.extraction_methods:
            try:
                # First try with lattice method (works better for bordered tables)
                tables_found = camelot.read_pdf(file_path, pages=pages, flavor=method)
                
                if len(tables_found) > 0:
                    results["tables_found"] = len(tables_found)
                    results["extraction_method"] = method
                    
                    # Calculate average accuracy score
                    accuracy_scores = [table.parsing_report['accuracy'] for table in tables_found]
                    results["extraction_score"] = sum(accuracy_scores) / len(accuracy_scores)
                    
                    # Convert camelot tables to pandas DataFrames
                    for table in tables_found:
                        # Clean the table data
                        df = self._clean_table(table.df)
                        
                        # Process the DataFrame
                        processed_df, processing_results = self.mapper.process_dataframe(df)
                        tables.append(processed_df)
                    
                    break
            except Exception as e:
                results["processing_issues"].append(f"Error with {method} method: {str(e)}")
        
        if not tables:
            # If no tables were found with camelot, try text extraction
            results["extraction_method"] = "text"
            try:
                text = extract_text(file_path)
                text_tables = self._extract_tables_from_text(text)
                
                for text_table in text_tables:
                    processed_df, processing_results = self.mapper.process_dataframe(text_table)
                    tables.append(processed_df)
                
                results["tables_found"] = len(tables)
            except Exception as e:
                results["processing_issues"].append(f"Error with text extraction: {str(e)}")
        
        return tables, results
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize a table extracted from PDF
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Use first row as header if it looks like a header
        if not any(str(col).isdigit() for col in df.columns):
            if df.shape[0] > 0:
                # Check if first row is different from other rows (likely a header)
                if self._is_header_row(df.iloc[0]):
                    df.columns = df.iloc[0]
                    df = df.iloc[1:].reset_index(drop=True)
        
        # Clean column names
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def _is_header_row(self, row: pd.Series) -> bool:
        """
        Check if a row is likely to be a header row
        
        Args:
            row: Row to check
            
        Returns:
            True if row is likely a header
        """
        # If many values in the row contain all caps or have different data types
        # from the rest of the table, it's likely a header
        all_caps_count = sum(1 for val in row if str(val).isupper())
        return all_caps_count > len(row) / 3
    
    def _clean_column_name(self, col_name: Any) -> str:
        """
        Clean a column name extracted from PDF
        
        Args:
            col_name: Column name to clean
            
        Returns:
            Cleaned column name
        """
        if pd.isna(col_name):
            return "Unknown"
        
        name = str(col_name).strip()
        # Replace newlines and excessive spaces
        name = re.sub(r'\s+', ' ', name)
        # Remove special characters
        name = re.sub(r'[^\w\s]', '', name)
        return name.strip()
    
    def _extract_tables_from_text(self, text: str) -> List[pd.DataFrame]:
        """
        Extract tables from plain text using heuristics
        
        Args:
            text: Extracted text from PDF
            
        Returns:
            List of DataFrames representing tables
        """
        tables = []
        
        # Split text into lines
        lines = text.split('\n')
        current_table_lines = []
        in_table = False
        
        for line in lines:
            # If the line has a table-like structure with multiple fields
            # separated by spaces or tabs
            fields = re.split(r'\s{2,}|\t', line.strip())
            
            if len(fields) >= 3 and any(field.strip() for field in fields):
                if not in_table:
                    in_table = True
                    current_table_lines = []
                    
                current_table_lines.append(fields)
            else:
                # End of a table section
                if in_table and len(current_table_lines) >= 2:
                    # Convert to dataframe
                    table_df = self._convert_text_lines_to_df(current_table_lines)
                    if table_df is not None and not table_df.empty:
                        tables.append(table_df)
                    
                    in_table = False
        
        # Don't forget the last table if text ends with a table
        if in_table and len(current_table_lines) >= 2:
            table_df = self._convert_text_lines_to_df(current_table_lines)
            if table_df is not None and not table_df.empty:
                tables.append(table_df)
        
        return tables
    
    def _convert_text_lines_to_df(self, lines: List[List[str]]) -> Optional[pd.DataFrame]:
        """
        Convert a list of tokenized lines to a DataFrame
        
        Args:
            lines: List of list of field values
            
        Returns:
            DataFrame or None if conversion fails
        """
        if not lines:
            return None
        
        # Standardize the number of columns
        max_columns = max(len(line) for line in lines)
        standardized_lines = []
        
        for line in lines:
            # Pad with empty strings if needed
            padded_line = line + [''] * (max_columns - len(line))
            standardized_lines.append(padded_line)
        
        # First line as header
        header = standardized_lines[0]
        data = standardized_lines[1:]
        
        # Clean header
        header = [self._clean_column_name(col) for col in header]
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=header)
        
        return df
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict with file information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        file_info = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_type": "pdf",
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        }
        
        try:
            # Try to extract the number of pages and text sample
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            file_info["page_count"] = len(doc)
            
            # Get a text sample from the first page
            if len(doc) > 0:
                text = doc[0].get_text()
                file_info["text_sample"] = text[:500] + "..." if len(text) > 500 else text
                
                # Try to detect if it's a tabular PDF
                tables = camelot.read_pdf(file_path, pages='1', flavor='lattice')
                file_info["has_tables"] = len(tables) > 0
                if len(tables) > 0:
                    file_info["table_count_first_page"] = len(tables)
            
            doc.close()
        except Exception as e:
            file_info["extraction_error"] = str(e)
        
        return file_info