"""
File management infrastructure for image downloads and file operations

This module provides utilities for downloading images, managing file paths,
and handling file system operations.
"""

import os
import re
import requests
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    """Configuration for file downloads"""
    timeout: int = 30
    chunk_size: int = 8192
    max_retries: int = 3


class FileManager:
    """Manages file operations and downloads"""
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        self.config = config or DownloadConfig()
    
    def ensure_directory(self, path: str) -> Path:
        """
        Ensure directory exists, creating it if necessary
        
        Args:
            path: Directory path to ensure exists
            
        Returns:
            Path object for the directory
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")
        return dir_path
    
    def generate_filename(
        self, 
        base_name: Optional[str] = None, 
        index: Optional[int] = None,
        extension: str = "png",
        timestamp: bool = True
    ) -> str:
        """
        Generate a unique filename
        
        Args:
            base_name: Base name for the file (optional)
            index: Index number to append (optional)  
            extension: File extension (default: png)
            timestamp: Whether to include timestamp (default: True)
            
        Returns:
            Generated filename
        """
        if not base_name:
            base_name = "tuzi_image"
        
        parts = [base_name]
        
        if index is not None:
            parts.append(str(index))
            
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(timestamp_str)
        
        filename = "_".join(parts) + f".{extension}"
        logger.debug(f"Generated filename: {filename}")
        return filename
    
    def download_file(self, url: str, filepath: Path) -> bool:
        """
        Download a file from URL to specified path
        
        Args:
            url: URL to download from
            filepath: Local path to save file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Downloading: {url} -> {filepath}")
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(
                    url, 
                    timeout=self.config.timeout,
                    stream=True
                )
                response.raise_for_status()
                
                # Write file in chunks
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            f.write(chunk)
                
                logger.info(f"Successfully downloaded: {filepath}")
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to download {url} after {self.config.max_retries} attempts")
                    return False
        
        return False
    
    def move_file(self, source: Path, destination: Path) -> bool:
        """
        Move file from source to destination
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source), str(destination))
            logger.info(f"Moved file: {source} -> {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move file {source} -> {destination}: {e}")
            return False
    
    def detect_file_extension(self, url: str, content_type: Optional[str] = None) -> str:
        """
        Detect appropriate file extension from URL or content type
        
        Args:
            url: File URL
            content_type: HTTP content type header (optional)
            
        Returns:
            File extension (without dot)
        """
        # First try to get extension from URL
        if url.endswith(('.jpg', '.jpeg')):
            return 'jpg'
        elif url.endswith('.webp'):
            return 'webp'
        elif url.endswith('.png'):
            return 'png'
        
        # Try content type
        if content_type:
            if 'jpeg' in content_type.lower():
                return 'jpg'
            elif 'webp' in content_type.lower():
                return 'webp'
            elif 'png' in content_type.lower():
                return 'png'
        
        # Default to png
        return 'png'


class ImageUrlExtractor:
    """Extracts image URLs from various response formats"""
    
    @staticmethod
    def extract_filesystem_urls(content: str) -> List[str]:
        """
        Extract filesystem.site image URLs from response content
        
        Args:
            content: Response content text
            
        Returns:
            List of image URLs
        """
        # Pattern to match filesystem.site URLs
        url_pattern = r'https://filesystem\.site/cdn/(?:download/)?(\d{8})/([a-zA-Z0-9]+)\.(?:png|jpg|jpeg|webp)'
        matches = re.findall(url_pattern, content)
        
        # Convert to full download URLs and remove duplicates
        download_urls = []
        seen_filenames = set()
        
        for date, filename in matches:
            if filename not in seen_filenames:
                # Try to detect format from content or default to png
                format_ext = "png"
                if "jpeg" in content.lower() or "jpg" in content.lower():
                    format_ext = "jpg"
                elif "webp" in content.lower():
                    format_ext = "webp"
                
                download_url = f"https://filesystem.site/cdn/download/{date}/{filename}.{format_ext}"
                download_urls.append(download_url)
                seen_filenames.add(filename)
        
        logger.info(f"Extracted {len(download_urls)} filesystem.site URLs")
        return download_urls
    
    @staticmethod
    def extract_flux_urls(result: Dict[str, Any]) -> List[str]:
        """
        Extract image URLs from FLUX API response
        
        Args:
            result: FLUX API response dictionary
            
        Returns:
            List of image URLs
        """
        try:
            if "data" in result and isinstance(result["data"], list):
                urls = []
                for item in result["data"]:
                    if "url" in item:
                        urls.append(item["url"])
                
                logger.info(f"Extracted {len(urls)} FLUX URLs")
                return urls
            else:
                logger.warning("No data field found in FLUX response")
                return []
        except Exception as e:
            logger.error(f"Error extracting FLUX URLs: {e}")
            return []


class BatchDownloader:
    """Handles batch downloading of multiple files"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    def download_images(
        self,
        urls: List[str],
        output_dir: str = "images",
        base_name: Optional[str] = None
    ) -> List[str]:
        """
        Download multiple images from URLs
        
        Args:
            urls: List of image URLs to download
            output_dir: Output directory for downloaded images
            base_name: Base name for generated filenames
            
        Returns:
            List of successfully downloaded file paths
        """
        if not urls:
            logger.warning("No URLs provided for download")
            return []
        
        # Ensure output directory exists
        output_path = self.file_manager.ensure_directory(output_dir)
        
        downloaded_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Starting batch download of {len(urls)} images")
        
        for i, url in enumerate(urls):
            try:
                # Generate filename
                filename = self.file_manager.generate_filename(
                    base_name=base_name,
                    index=i + 1,
                    extension=self.file_manager.detect_file_extension(url),
                    timestamp=True
                )
                
                filepath = output_path / filename
                
                # Download the file
                if self.file_manager.download_file(url, filepath):
                    downloaded_files.append(str(filepath))
                
            except Exception as e:
                logger.error(f"Failed to download image {i+1}/{len(urls)} from {url}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(downloaded_files)}/{len(urls)} images")
        return downloaded_files