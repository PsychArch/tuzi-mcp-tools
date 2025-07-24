"""
API client infrastructure for Tu-zi.com API

This module provides the core HTTP client functionality for interacting
with the Tu-zi.com API, including authentication, request handling, and
response processing.
"""

import aiohttp
import requests
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ApiConfig:
    """Configuration for API client"""
    api_key: str
    base_url: str = "https://api.tu-zi.com/v1"
    timeout: int = 300  # 5 minutes
    
    def get_headers(self) -> Dict[str, str]:
        """Get standard headers for API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }


class TuZiApiClient:
    """HTTP client for Tu-zi.com API with sync and async support"""
    
    def __init__(self, config: ApiConfig):
        self.config = config
        
    def post_sync(
        self, 
        endpoint: str, 
        data: Dict[str, Any], 
        stream: bool = False
    ) -> requests.Response:
        """
        Make synchronous POST request
        
        Args:
            endpoint: API endpoint (e.g., "chat/completions")
            data: Request payload
            stream: Whether to use streaming response
            
        Returns:
            requests.Response object
            
        Raises:
            Exception: If the request fails
        """
        url = f"{self.config.base_url}/{endpoint}"
        headers = self.config.get_headers()
        
        logger.info(f"Making sync POST request to {endpoint}")
        logger.debug(f"Request URL: {url}")
        logger.debug(f"Request payload: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            url,
            json=data,
            headers=headers,
            timeout=self.config.timeout,
            stream=stream
        )
        
        if response.status_code != 200:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            logger.error(f"Request URL: {url}")
            logger.error(f"Request payload: {json.dumps(data, indent=2)}")
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        logger.debug(f"API request successful: {response.status_code}")
            
        return response
    
    async def post_async(
        self, 
        endpoint: str, 
        data: Dict[str, Any],
        session: Optional[aiohttp.ClientSession] = None
    ) -> aiohttp.ClientResponse:
        """
        Make asynchronous POST request
        
        Args:
            endpoint: API endpoint (e.g., "chat/completions")
            data: Request payload
            session: Optional existing session to use
            
        Returns:
            aiohttp.ClientResponse object
            
        Raises:
            Exception: If the request fails
        """
        url = f"{self.config.base_url}/{endpoint}"
        headers = self.config.get_headers()
        
        logger.info(f"Making async POST request to {endpoint}")
        logger.debug(f"Request URL: {url}")
        logger.debug(f"Request payload: {json.dumps(data, indent=2)}")
        
        # Use provided session or create new one
        if session:
            response = await session.post(
                url,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"API request failed: {response.status} - {error_text}")
                logger.error(f"Request URL: {url}")
                logger.error(f"Request payload: {json.dumps(data, indent=2)}")
                raise Exception(f"API Error: {response.status} - {error_text}")
            
            logger.debug(f"API request successful: {response.status}")
                
            return response
        else:
            async with aiohttp.ClientSession() as new_session:
                return await self.post_async(endpoint, data, new_session)
    
    async def post_async_with_stream_processing(
        self, 
        endpoint: str, 
        data: Dict[str, Any],
        session: Optional[aiohttp.ClientSession] = None
    ) -> Dict[str, Any]:
        """
        Make asynchronous POST request and process streaming response within session context
        
        Args:
            endpoint: API endpoint (e.g., "chat/completions")
            data: Request payload
            session: Optional existing session to use
            
        Returns:
            Dictionary with processed result and full content
            
        Raises:
            Exception: If the request fails
        """
        url = f"{self.config.base_url}/{endpoint}"
        headers = self.config.get_headers()
        
        logger.info(f"Making async POST request with stream processing to {endpoint}")
        logger.debug(f"Request URL: {url}")
        logger.debug(f"Request payload: {json.dumps(data, indent=2)}")
        
        # Use provided session or create new one
        if session:
            async with session.post(
                url,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    logger.error(f"Request URL: {url}")
                    logger.error(f"Request payload: {json.dumps(data, indent=2)}")
                    raise Exception(f"API Error: {response.status} - {error_text}")
                
                logger.debug(f"API request successful: {response.status}")
                
                # Process stream within session context
                return await StreamProcessor.process_async_stream(response)
        else:
            async with aiohttp.ClientSession() as new_session:
                return await self.post_async_with_stream_processing(endpoint, data, new_session)


class StreamProcessor:
    """Utility class for processing streaming API responses"""
    
    @staticmethod
    def process_sync_stream(response: requests.Response) -> Dict[str, Any]:
        """
        Process synchronous streaming response
        
        Args:
            response: requests.Response object with stream=True
            
        Returns:
            Dictionary with processed result and full content
        """
        full_content = ""
        result = {}
        
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # Remove 'data: ' prefix if present
                if line.startswith(b'data: '):
                    line = line[6:]
                
                # Skip keep-alive lines
                if line == b'[DONE]':
                    break
                    
                try:
                    data = json.loads(line)
                    
                    # Extract content from streaming response
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            full_content += content
                    
                    # Store the last received data as the result
                    result = data
                    
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing sync stream: {e}")
            raise
        
        return {
            "result": result,
            "content": full_content
        }
    
    @staticmethod
    async def process_async_stream(response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """
        Process asynchronous streaming response
        
        Args:
            response: aiohttp.ClientResponse object
            
        Returns:
            Dictionary with processed result and full content
        """
        full_content = ""
        result = {}
        
        try:
            # Read the response properly using iter_chunked or iter_content
            buffer = b""
            async for chunk in response.content.iter_chunked(1024):
                if not chunk:
                    break
                
                buffer += chunk
                
                # Process complete lines
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    
                    if not line.strip():
                        continue
                    
                    try:
                        line_str = line.decode('utf-8').strip()
                        
                        # Remove 'data: ' prefix if present
                        if line_str.startswith('data: '):
                            line_str = line_str[6:]
                        
                        # Skip keep-alive lines
                        if line_str == '[DONE]':
                            break
                        
                        # Skip empty lines
                        if not line_str:
                            continue
                            
                        try:
                            data = json.loads(line_str)
                            
                            # Extract content from streaming response
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    full_content += content
                            
                            # Store the last received data as the result
                            result = data
                            
                        except json.JSONDecodeError as json_e:
                            logger.debug(f"JSON decode error for line: {line_str[:100]}...")
                            continue
                            
                    except UnicodeDecodeError as decode_e:
                        logger.debug(f"Unicode decode error: {decode_e}")
                        continue
                    
        except Exception as e:
            logger.error(f"Error processing async stream: {e}")
            logger.error(f"Stream content read so far: {len(full_content)} characters")
            logger.error(f"Last result: {result}")
            raise
        
        return {
            "result": result,
            "content": full_content
        }


def validate_api_response(response_data: Dict[str, Any]) -> None:
    """
    Validate API response for errors
    
    Args:
        response_data: Parsed JSON response from API
        
    Raises:
        Exception: If the response contains an error
    """
    if "error" in response_data:
        error_msg = response_data["error"].get("message", "Unknown API error")
        logger.error(f"API returned error: {error_msg}")
        raise Exception(f"API Error: {error_msg}")