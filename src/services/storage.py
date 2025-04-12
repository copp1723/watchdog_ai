"""
Supabase storage service for Watchdog AI.

This module handles file storage and retrieval from Supabase.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, BinaryIO
import tempfile
from supabase import create_client, Client

# Import configuration
from config.config import (
    SUPABASE_URL, SUPABASE_KEY, UPLOAD_BUCKET_NAME, 
    PARSED_DATA_BUCKET_NAME, DB_CONFIG
)


class StorageService:
    """Service for storing and retrieving files and data from Supabase."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize the storage service.
        
        Args:
            url: Optional Supabase URL (uses config if not provided)
            key: Optional Supabase key (uses config if not provided)
        """
        self.url = url or SUPABASE_URL
        self.key = key or SUPABASE_KEY
        self._client = None
        
        self.upload_bucket = DB_CONFIG.get("upload_bucket", UPLOAD_BUCKET_NAME)
        self.parsed_data_bucket = DB_CONFIG.get("parsed_data_bucket", PARSED_DATA_BUCKET_NAME)
    
    @property
    def client(self) -> Client:
        """
        Get the Supabase client, initializing it if necessary.
        
        Returns:
            Client: Supabase client
        """
        if self._client is None:
            self._client = create_client(self.url, self.key)
        return self._client
    
    def ensure_buckets_exist(self) -> Tuple[bool, Optional[str]]:
        """
        Ensure that the necessary storage buckets exist, creating them if needed.
        
        Returns:
            tuple: (success, error_message_if_any)
        """
        try:
            # Get list of existing buckets
            existing_buckets = self.client.storage.list_buckets()
            existing_names = [bucket["name"] for bucket in existing_buckets]
            
            # Create upload bucket if it doesn't exist
            if self.upload_bucket not in existing_names:
                self.client.storage.create_bucket(self.upload_bucket, {"public": False})
            
            # Create parsed data bucket if it doesn't exist
            if self.parsed_data_bucket not in existing_names:
                self.client.storage.create_bucket(self.parsed_data_bucket, {"public": False})
            
            return True, None
            
        except Exception as e:
            return False, f"Error ensuring buckets exist: {str(e)}"
    
    def upload_file(self, file_path: str, destination_path: Optional[str] = None, 
                   bucket_name: Optional[str] = None) -> Tuple[bool, Union[str, Dict]]:
        """
        Upload a file to Supabase storage.
        
        Args:
            file_path: Path to the file to upload
            destination_path: Path in the bucket (defaults to filename)
            bucket_name: Bucket to upload to (defaults to upload_bucket)
            
        Returns:
            tuple: (success, result_or_error_message)
        """
        try:
            # Ensure buckets exist
            success, error = self.ensure_buckets_exist()
            if not success:
                return False, error
            
            # Default values
            bucket_name = bucket_name or self.upload_bucket
            file_name = Path(file_path).name
            destination_path = destination_path or file_name
            
            # Upload the file
            with open(file_path, 'rb') as f:
                result = self.client.storage.from_(bucket_name).upload(
                    destination_path, f, {"content-type": "application/octet-stream"}
                )
            
            return True, result
            
        except Exception as e:
            return False, f"Error uploading file: {str(e)}"
    
    def upload_content(self, content: Union[bytes, BinaryIO], destination_path: str,
                      bucket_name: Optional[str] = None) -> Tuple[bool, Union[str, Dict]]:
        """
        Upload content directly to Supabase storage.
        
        Args:
            content: Content to upload (bytes or file-like object)
            destination_path: Path in the bucket
            bucket_name: Bucket to upload to (defaults to upload_bucket)
            
        Returns:
            tuple: (success, result_or_error_message)
        """
        try:
            # Ensure buckets exist
            success, error = self.ensure_buckets_exist()
            if not success:
                return False, error
            
            # Default bucket
            bucket_name = bucket_name or self.upload_bucket
            
            # Upload the content
            result = self.client.storage.from_(bucket_name).upload(
                destination_path, content, {"content-type": "application/octet-stream"}
            )
            
            return True, result
            
        except Exception as e:
            return False, f"Error uploading content: {str(e)}"
    
    def download_file(self, file_path: str, bucket_name: Optional[str] = None) -> Tuple[bool, Union[bytes, str]]:
        """
        Download a file from Supabase storage.
        
        Args:
            file_path: Path in the bucket
            bucket_name: Bucket to download from (defaults to upload_bucket)
            
        Returns:
            tuple: (success, content_or_error_message)
        """
        try:
            # Default bucket
            bucket_name = bucket_name or self.upload_bucket
            
            # Download the file
            content = self.client.storage.from_(bucket_name).download(file_path)
            
            return True, content
            
        except Exception as e:
            return False, f"Error downloading file: {str(e)}"
    
    def list_files(self, path: str = "", bucket_name: Optional[str] = None) -> Tuple[bool, Union[List, str]]:
        """
        List files in a bucket.
        
        Args:
            path: Path prefix to filter by
            bucket_name: Bucket to list (defaults to upload_bucket)
            
        Returns:
            tuple: (success, file_list_or_error_message)
        """
        try:
            # Default bucket
            bucket_name = bucket_name or self.upload_bucket
            
            # List files
            result = self.client.storage.from_(bucket_name).list(path)
            
            return True, result
            
        except Exception as e:
            return False, f"Error listing files: {str(e)}"
    
    def delete_file(self, file_path: str, bucket_name: Optional[str] = None) -> Tuple[bool, Union[Dict, str]]:
        """
        Delete a file from Supabase storage.
        
        Args:
            file_path: Path in the bucket
            bucket_name: Bucket to delete from (defaults to upload_bucket)
            
        Returns:
            tuple: (success, result_or_error_message)
        """
        try:
            # Default bucket
            bucket_name = bucket_name or self.upload_bucket
            
            # Delete the file
            result = self.client.storage.from_(bucket_name).remove([file_path])
            
            return True, result
            
        except Exception as e:
            return False, f"Error deleting file: {str(e)}"
    
    def store_parsed_data(self, parsed_data: Dict[str, Any], file_identifier: str) -> Tuple[bool, Union[str, Dict]]:
        """
        Store parsed data in the parsed_data bucket.
        
        Args:
            parsed_data: The parsed data to store
            file_identifier: Identifier for the data (original filename or UUID)
            
        Returns:
            tuple: (success, result_or_error_message)
        """
        try:
            # Ensure buckets exist
            success, error = self.ensure_buckets_exist()
            if not success:
                return False, error
            
            # Create a JSON string from the parsed data
            # Handle non-serializable types like dates and numpy values
            json_str = json.dumps(parsed_data, default=str)
            
            # Create a destination path with timestamp to avoid collisions
            timestamp = int(time.time())
            destination_path = f"{file_identifier.split('.')[0]}_{timestamp}.json"
            
            # Upload the JSON data
            result = self.client.storage.from_(self.parsed_data_bucket).upload(
                destination_path, json_str.encode('utf-8'), {"content-type": "application/json"}
            )
            
            return True, destination_path
            
        except Exception as e:
            return False, f"Error storing parsed data: {str(e)}"
    
    def get_parsed_data(self, file_path: str) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Get parsed data from the parsed_data bucket.
        
        Args:
            file_path: Path in the bucket
            
        Returns:
            tuple: (success, data_or_error_message)
        """
        try:
            # Download the file
            success, content = self.download_file(file_path, self.parsed_data_bucket)
            if not success:
                return False, content
            
            # Parse the JSON
            parsed = json.loads(content.decode('utf-8'))
            
            return True, parsed
            
        except Exception as e:
            return False, f"Error getting parsed data: {str(e)}"
    
    def list_parsed_data(self, prefix: str = "") -> Tuple[bool, Union[List[Dict[str, Any]], str]]:
        """
        List all parsed data files.
        
        Args:
            prefix: Optional prefix to filter by
            
        Returns:
            tuple: (success, file_list_or_error_message)
        """
        try:
            # List files
            success, files = self.list_files(prefix, self.parsed_data_bucket)
            if not success:
                return False, files
            
            # Return file list
            return True, files
            
        except Exception as e:
            return False, f"Error listing parsed data: {str(e)}"


if __name__ == "__main__":
    # Example usage
    storage = StorageService()
    
    # Test ensuring buckets exist
    success, result = storage.ensure_buckets_exist()
    if success:
        print("Buckets exist or were created successfully")
    else:
        print(f"Error: {result}")
    
    # Test uploading a file
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
    #     tmp.write(b"Test content")
    #     tmp_path = tmp.name
    # 
    # success, result = storage.upload_file(tmp_path)
    # if success:
    #     print(f"File uploaded successfully: {result}")
    # else:
    #     print(f"Error: {result}")
    # 
    # # Clean up
    # os.remove(tmp_path)