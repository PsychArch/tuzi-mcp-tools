"""
Core functionality for Tuzi Image Generator

This module contains the main TuZiImageGenerator class and utilities
that are shared between CLI and MCP interfaces.
"""

import os
import requests
import json
import time
import re
import logging
import sys
import threading
import uuid
import asyncio
import aiohttp
import tempfile
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from collections import OrderedDict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.markdown import Markdown
from rich.live import Live

# Configure logging to stderr for MCP server compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Model order from lowest to highest price (fallback order)
MODEL_FALLBACK_ORDER = [
    "gpt-image-1",      # $0.04
    "gpt-4o-image",     # $0.04  
    "gpt-4o-image-vip", # $0.10
    "gpt-image-1-vip"   # $0.10
]

# Concurrent limits configuration - configurable via environment variable
CONCURRENT_LIMIT = int(os.getenv("TUZI_CONCURRENT_LIMIT", "60"))

# Configuration options
QUALITY_OPTIONS = ["low", "medium", "high", "auto"]
SIZE_OPTIONS = ["1024x1024", "1536x1024", "1024x1536", "auto"]
FORMAT_OPTIONS = ["png", "jpeg", "webp"]
BACKGROUND_OPTIONS = ["transparent", "opaque"]

# FLUX-specific configuration options
FLUX_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"]
FLUX_OUTPUT_FORMATS = ["png", "jpg", "jpeg", "webp"]


class AsyncTaskManager:
    """
    Global async task manager for MCP server with LRU cache capabilities.
    Maintains task state across MCP tool calls in persistent server.
    """
    
    def __init__(self, max_completed_tasks: int = 100):
        self.max_completed_tasks = max_completed_tasks
        self._lock = asyncio.Lock()
        
        # Task storage with LRU behavior
        self.executing_tasks: Dict[str, asyncio.Task] = {}  # task_id -> asyncio.Task
        self.completed_tasks: OrderedDict[str, Dict[str, Any]] = OrderedDict()  # task_id -> result
        
        logger.info(f"AsyncTaskManager initialized with max_completed_tasks={max_completed_tasks}")
    
    async def submit_task(self, task_id: str, coro) -> str:
        """
        Submit an async task for execution and return task ID immediately.
        
        Args:
            task_id: Unique task identifier
            coro: Coroutine to execute
            
        Returns:
            Task ID for tracking
        """
        async with self._lock:
            # Create and start the async task
            task = asyncio.create_task(self._execute_task(task_id, coro))
            self.executing_tasks[task_id] = task
            
            logger.info(f"Task {task_id} submitted and started execution")
            return task_id
    
    async def _execute_task(self, task_id: str, coro) -> None:
        """
        Internal method to execute a task and store its result.
        """
        try:
            # Execute the coroutine
            result = await coro
            
            # Store the result
            async with self._lock:
                # Remove from executing tasks
                if task_id in self.executing_tasks:
                    del self.executing_tasks[task_id]
                
                # Add to completed tasks with LRU management
                self.completed_tasks[task_id] = {
                    "success": True,
                    "result": result,
                    "error": None,
                    "completed_at": datetime.now().isoformat()
                }
                
                # LRU cleanup - remove oldest completed tasks if over limit
                while len(self.completed_tasks) > self.max_completed_tasks:
                    oldest_task_id = next(iter(self.completed_tasks))
                    logger.info(f"LRU cleanup: removing oldest completed task {oldest_task_id}")
                    del self.completed_tasks[oldest_task_id]
                
                logger.info(f"Task {task_id} completed successfully")
                
        except Exception as e:
            # Store the error
            async with self._lock:
                # Remove from executing tasks
                if task_id in self.executing_tasks:
                    del self.executing_tasks[task_id]
                
                # Add error to completed tasks
                self.completed_tasks[task_id] = {
                    "success": False,
                    "result": None,
                    "error": str(e),
                    "completed_at": datetime.now().isoformat()
                }
                
                # LRU cleanup
                while len(self.completed_tasks) > self.max_completed_tasks:
                    oldest_task_id = next(iter(self.completed_tasks))
                    del self.completed_tasks[oldest_task_id]
                
                logger.error(f"Task {task_id} failed: {e}")
    
    async def wait_for_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Wait for all executing tasks to complete and return all results.
        
        Returns:
            Dictionary mapping task_ids to their results
        """
        # Get snapshot of currently executing tasks
        async with self._lock:
            executing_tasks_snapshot = dict(self.executing_tasks)
            logger.info(f"Waiting for {len(executing_tasks_snapshot)} executing tasks")
        
        # Wait for all executing tasks to complete
        if executing_tasks_snapshot:
            try:
                # Wait for all tasks with a reasonable timeout
                await asyncio.wait_for(
                    asyncio.gather(*executing_tasks_snapshot.values(), return_exceptions=True),
                    timeout=600  # 10 minutes total timeout
                )
                logger.info("All executing tasks completed")
            except asyncio.TimeoutError:
                logger.warning("Some tasks timed out during barrier wait")
        
        # Collect all completed results
        async with self._lock:
            results = dict(self.completed_tasks)
            # Clear completed tasks after collecting (barrier cleans up)
            self.completed_tasks.clear()
            logger.info(f"Barrier returning {len(results)} task results")
            
        return results
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current task manager status"""
        async with self._lock:
            return {
                "executing_tasks": len(self.executing_tasks),
                "completed_tasks": len(self.completed_tasks),
                "executing_task_ids": list(self.executing_tasks.keys()),
                "completed_task_ids": list(self.completed_tasks.keys())
            }


class RollingGroup:
    """
    Async task group that allows submitting tasks and waiting for all submitted tasks to complete.
    This enables a submit/barrier pattern for concurrent image generation.
    """
    
    def __init__(self) -> None:
        self._tg = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    async def submit(self, coro):
        """Submit a coroutine to be executed concurrently"""
        async with self._lock:
            # Lazily open a group on first use
            if self._tg is None:
                self._tg = asyncio.TaskGroup()
                await self._tg.__aenter__()
            
            # Wrap the coroutine with semaphore to limit concurrency
            async def _limited_coro():
                async with self._semaphore:
                    return await coro
            
            self._tg.create_task(_limited_coro())

    async def barrier(self):
        """Wait for all submitted tasks to complete and return their results"""
        async with self._lock:
            tg, self._tg = self._tg, None     # detach current group
        if tg is not None:                    # close outside the lock
            await tg.__aexit__(None, None, None)


class ConversationManager:
    """Manages conversation history for both CLI and MCP modes with automatic ID generation and circular buffer"""
    
    def __init__(self, storage_mode: str = "memory"):
        """
        Initialize the conversation manager
        
        Args:
            storage_mode: "memory" for in-memory storage (MCP), "file" for file storage (CLI)
        """
        self.storage_mode = storage_mode
        self.conversation_dir = Path.cwd()  # Default to current directory for file mode
        
        # Get circular buffer depth from environment variable (default: 10)
        self.max_conversations = int(os.getenv("TUZI_CONVERSATION_BUFFER_SIZE", "10"))
        
        # Use OrderedDict for LRU-style circular buffer in memory mode
        self.conversations = OrderedDict()  # In-memory storage for MCP mode
        
        # Counter for generating sequential conversation IDs
        self._id_counter = 1
        self._id_lock = threading.Lock()
        
    def _validate_conversation_id(self, conversation_id: str) -> bool:
        """Validate conversation ID format (alphanumeric + hyphens/underscores)"""
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', conversation_id))
    
    def _get_conversation_file_path(self, conversation_id: str, conversation_type: str) -> Path:
        """Get the file path for a conversation"""
        filename = f"tuzi-{conversation_type}-{conversation_id}.json"
        return self.conversation_dir / filename
    
    def generate_conversation_id(self, conversation_type: str) -> str:
        """
        Generate a unique conversation ID automatically
        
        Args:
            conversation_type: Type of conversation ("survey", "image", "flux")
            
        Returns:
            Generated conversation ID
        """
        with self._id_lock:
            # Generate ID with format: {type}_{counter}
            conversation_id = f"{conversation_type}_{self._id_counter}"
            self._id_counter += 1
            return conversation_id
    
    def load_conversation(self, conversation_id: str, conversation_type: str) -> List[Dict[str, Any]]:
        """
        Load conversation history
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_type: Type of conversation ("survey", "image", "flux")
            
        Returns:
            List of message dictionaries
        """
        if not self._validate_conversation_id(conversation_id):
            raise ValueError(f"Invalid conversation_id format: {conversation_id}")
        
        conversation_key = f"{conversation_type}:{conversation_id}"
        
        if self.storage_mode == "memory":
            # Move to end (mark as recently accessed) if exists
            if conversation_key in self.conversations:
                self.conversations.move_to_end(conversation_key)
                return self.conversations[conversation_key]
            return []
        
        elif self.storage_mode == "file":
            file_path = self._get_conversation_file_path(conversation_id, conversation_type)
            
            if not file_path.exists():
                return []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('messages', [])
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load conversation {conversation_id}: {e}")
                return []
        
        return []
    
    def save_conversation(self, conversation_id: str, conversation_type: str, messages: List[Dict[str, Any]]) -> None:
        """
        Save conversation history with circular buffer management
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_type: Type of conversation ("survey", "image", "flux")
            messages: List of message dictionaries
        """
        if not self._validate_conversation_id(conversation_id):
            raise ValueError(f"Invalid conversation_id format: {conversation_id}")
        
        conversation_key = f"{conversation_type}:{conversation_id}"
        
        if self.storage_mode == "memory":
            # Add/update conversation
            self.conversations[conversation_key] = messages
            # Move to end (mark as most recently used)
            self.conversations.move_to_end(conversation_key)
            
            # Implement circular buffer: remove oldest if over limit
            while len(self.conversations) > self.max_conversations:
                oldest_key = next(iter(self.conversations))
                logger.info(f"Removing oldest conversation from buffer: {oldest_key}")
                del self.conversations[oldest_key]
        
        elif self.storage_mode == "file":
            file_path = self._get_conversation_file_path(conversation_id, conversation_type)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            conversation_data = {
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
                "created_at": datetime.now().isoformat() if not file_path.exists() else None,
                "updated_at": datetime.now().isoformat(),
                "messages": messages
            }
            
            # If file exists, preserve created_at timestamp
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        conversation_data["created_at"] = existing_data.get("created_at")
                except (json.JSONDecodeError, IOError):
                    pass
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            except IOError as e:
                logger.error(f"Failed to save conversation {conversation_id}: {e}")
                raise
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        Get current buffer status information
        
        Returns:
            Dictionary with buffer status information
        """
        if self.storage_mode == "memory":
            return {
                "current_conversations": len(self.conversations),
                "max_conversations": self.max_conversations,
                "buffer_full": len(self.conversations) >= self.max_conversations,
                "conversation_keys": list(self.conversations.keys())
            }
        else:
            return {
                "storage_mode": "file",
                "max_conversations": "unlimited",
                "conversation_dir": str(self.conversation_dir)
            }
    
    def add_message(self, conversation_id: str, conversation_type: str, role: str, content: str) -> None:
        """
        Add a message to the conversation
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_type: Type of conversation ("survey", "image", "flux")
            role: Message role ("user" or "assistant")
            content: Message content
        """
        messages = self.load_conversation(conversation_id, conversation_type)
        
        # For survey conversations, filter out thinking content from assistant messages
        if conversation_type == "survey" and role == "assistant":
            content = self._filter_thinking_content(content)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        messages.append(message)
        self.save_conversation(conversation_id, conversation_type, messages)
    
    def get_conversation_summary(self, conversation_id: str, conversation_type: str) -> Dict[str, Any]:
        """
        Get conversation summary information
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_type: Type of conversation ("survey", "image", "flux")
            
        Returns:
            Dictionary with conversation summary
        """
        messages = self.load_conversation(conversation_id, conversation_type)
        
        if not messages:
            return {
                "exists": False,
                "message_count": 0,
                "created_at": None,
                "updated_at": None
            }
        
        # Try to get timestamps from file metadata if available
        created_at = None
        updated_at = None
        
        if self.storage_mode == "file":
            file_path = self._get_conversation_file_path(conversation_id, conversation_type)
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        created_at = data.get("created_at")
                        updated_at = data.get("updated_at")
                except (json.JSONDecodeError, IOError):
                    pass
        
        return {
            "exists": True,
            "message_count": len(messages),
            "created_at": created_at,
            "updated_at": updated_at,
            "last_user_message": next((msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"] 
                                      for msg in reversed(messages) if msg["role"] == "user"), None)
        }
    
    def _filter_thinking_content(self, content: str) -> str:
        """
        Filter out thinking content from survey responses to keep only the final answer
        
        Args:
            content: Raw content from survey response
            
        Returns:
            Content with thinking sections removed
        """
        import re
        
        # The actual separator pattern from o3-all responses: "*Thought for X seconds*"
        # This can be seconds, minutes and seconds (like "1m 29s"), etc.
        thought_pattern = r'\*Thought for [^*]+\*'
        
        # Split content on the thinking separator
        parts = re.split(thought_pattern, content, maxsplit=1)
        
        if len(parts) > 1:
            # Found the separator, return content after it
            final_answer = parts[1].strip()
            if final_answer:
                return final_answer
        
        # If no separator found, return original content
        # (No fallback assumptions - only use what we know exists)
        return content


class TuZiImageGenerator:
    """Main class for generating images using Tu-zi.com API"""
    
    def __init__(self, api_key: str, console: Optional[Console] = None, conversation_manager: Optional[ConversationManager] = None):
        """
        Initialize the TuZi Image Generator
        
        Args:
            api_key: Tu-zi.com API key
            console: Rich console for output (optional)
            conversation_manager: ConversationManager instance for handling conversation history (optional)
        """
        self.api_key = api_key
        self.api_url = "https://api.tu-zi.com/v1/chat/completions"
        self.console = console or Console()
        self.conversation_manager = conversation_manager
        self._rolling_group = RollingGroup()
        self._task_results = []
    
    def generate_image(
        self, 
        prompt: str, 
        stream: bool = True,
        conversation_id: Optional[str] = None,
        input_image: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image using Tu-zi.com API with automatic model fallback
        
        Args:
            prompt: The image generation prompt
            stream: Whether to use streaming response
            conversation_id: Optional conversation ID for maintaining history (auto-generated if None)
            input_image: Base64 encoded reference image (optional)
            **kwargs: Additional parameters (quality, size, format, etc.)
            
        Returns:
            Dictionary containing the API response, model used, and conversation info
        """
        
        # Handle conversation management with auto-generated ID
        if conversation_id is None and self.conversation_manager:
            conversation_id = self.conversation_manager.generate_conversation_id("image")
        
        conversation_info = {"conversation_id": conversation_id, "conversation_continued": False}
        
        # Build messages list starting with conversation history
        messages = []
        
        # Load conversation history if conversation_id is provided
        if conversation_id and self.conversation_manager:
            try:
                history = self.conversation_manager.load_conversation(conversation_id, "image")
                messages.extend(history)
                if history:
                    conversation_info["conversation_continued"] = True
                    if self.console:
                        self.console.print(f"[blue]📖 Loaded conversation {conversation_id} with {len(history)} messages[/blue]")
            except Exception as e:
                logger.warning(f"Failed to load conversation {conversation_id}: {e}")
                # Continue without history
        
        # Build the current message content with image generation parameters
        if input_image:
            # Use multimodal format with image and text
            content_parts = [
                {"type": "text", "text": prompt}
            ]
            
            # Add image generation parameters if provided
            if any(k in kwargs for k in ['quality', 'size', 'format', 'background', 'output_compression']):
                params = []
                if 'quality' in kwargs and kwargs['quality'] != 'auto':
                    params.append(f"quality: {kwargs['quality']}")
                if 'size' in kwargs and kwargs['size'] != 'auto':
                    params.append(f"size: {kwargs['size']}")
                if 'format' in kwargs and kwargs['format'] != 'png':
                    params.append(f"format: {kwargs['format']}")
                if 'background' in kwargs and kwargs['background'] == 'transparent':
                    params.append("background: transparent")
                if 'output_compression' in kwargs:
                    params.append(f"compression: {kwargs['output_compression']}")
                
                if params:
                    content_parts[0]["text"] += f"\n\nImage parameters: {', '.join(params)}"
            
            # Add the input image
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": input_image
                }
            })
            
            # Add current user message to the conversation
            messages.append({
                "role": "user",
                "content": content_parts
            })
        else:
            # Use text-only format
            content = prompt
            
            # Add image generation parameters if provided
            if any(k in kwargs for k in ['quality', 'size', 'format', 'background', 'output_compression']):
                params = []
                if 'quality' in kwargs and kwargs['quality'] != 'auto':
                    params.append(f"quality: {kwargs['quality']}")
                if 'size' in kwargs and kwargs['size'] != 'auto':
                    params.append(f"size: {kwargs['size']}")
                if 'format' in kwargs and kwargs['format'] != 'png':
                    params.append(f"format: {kwargs['format']}")
                if 'background' in kwargs and kwargs['background'] == 'transparent':
                    params.append("background: transparent")
                if 'output_compression' in kwargs:
                    params.append(f"compression: {kwargs['output_compression']}")
                
                if params:
                    content += f"\n\nImage parameters: {', '.join(params)}"
            
            # Add current user message to the conversation
            messages.append({
                "role": "user",
                "content": content
            })
        
        # Try models in order from lowest to highest price
        last_exception = None
        
        for model in MODEL_FALLBACK_ORDER:
            try:
                # Log to stderr for debugging (works in MCP server)
                logger.info(f"Trying model: {model}")
                
                # Also display in console if available (CLI mode)
                if self.console:
                    self.console.print(f"[dim]🤖 Trying model: {model}[/dim]")
                
                data = {
                    "model": model,
                    "stream": stream,
                    "messages": messages
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                if stream:
                    # For streaming response
                    with requests.post(
                        self.api_url, 
                        json=data, 
                        headers=headers, 
                        timeout=300,  # 5 minutes timeout
                        stream=True
                    ) as response:
                        
                        if response.status_code != 200:
                            raise Exception(f"API Error: {response.status_code} - {response.text}")
                        
                        # Process the streaming response
                        result = self._process_stream(response)
                        # Add the successful model to the result
                        result["model_used"] = model
                        result["conversation_info"] = conversation_info
                        
                        # Save conversation if conversation_id is provided
                        if conversation_id and self.conversation_manager:
                            try:
                                # Extract assistant response from result
                                assistant_content = ""
                                if "choices" in result and len(result["choices"]) > 0:
                                    assistant_content = result["choices"][0].get("message", {}).get("content", "")
                                elif "content" in result:
                                    assistant_content = result["content"]
                                
                                if assistant_content:
                                    # Add assistant message to conversation
                                    messages.append({
                                        "role": "assistant",
                                        "content": assistant_content
                                    })
                                    
                                    # Save updated conversation
                                    self.conversation_manager.save_conversation(conversation_id, "image", messages)
                                    
                                    if self.console:
                                        self.console.print(f"[green]💾 Conversation {conversation_id} saved[/green]")
                            except Exception as e:
                                logger.warning(f"Failed to save conversation {conversation_id}: {e}")
                                # Don't fail the entire operation for conversation save errors
                        
                        # Log success to stderr
                        logger.info(f"Successfully generated with model: {model}")
                        
                        # Also display in console if available (CLI mode)
                        if self.console:
                            self.console.print(f"[green]✅ Successfully generated with model: {model}[/green]")
                        return result
                else:
                    # For non-streaming response
                    response = requests.post(
                        self.api_url, 
                        json=data, 
                        headers=headers, 
                        timeout=300  # 5 minutes timeout
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"API Error: {response.status_code} - {response.text}")
                    
                    result = response.json()
                    
                    if "error" in result:
                        raise Exception(f"API Error: {result['error']['message']}")
                    
                    # Add the successful model to the result
                    result["model_used"] = model
                    result["conversation_info"] = conversation_info
                    
                    # Save conversation if conversation_id is provided
                    if conversation_id and self.conversation_manager:
                        try:
                            # Extract assistant response from result
                            assistant_content = ""
                            if "choices" in result and len(result["choices"]) > 0:
                                assistant_content = result["choices"][0].get("message", {}).get("content", "")
                            elif "content" in result:
                                assistant_content = result["content"]
                            
                            if assistant_content:
                                # Add assistant message to conversation
                                messages.append({
                                    "role": "assistant",
                                    "content": assistant_content
                                })
                                
                                # Save updated conversation
                                self.conversation_manager.save_conversation(conversation_id, "image", messages)
                                
                                if self.console:
                                    self.console.print(f"[green]💾 Conversation {conversation_id} saved[/green]")
                        except Exception as e:
                            logger.warning(f"Failed to save conversation {conversation_id}: {e}")
                            # Don't fail the entire operation for conversation save errors
                    
                    # Log success to stderr
                    logger.info(f"Successfully generated with model: {model}")
                    
                    # Also display in console if available (CLI mode)
                    if self.console:
                        self.console.print(f"[green]✅ Successfully generated with model: {model}[/green]")
                    return result
                    
            except Exception as e:
                last_exception = e
                
                # Log failure to stderr
                logger.warning(f"Model {model} failed: {e}")
                
                # Also display in console if available (CLI mode)
                if self.console:
                    self.console.print(f"[yellow]⚠️ Model {model} failed: {e}[/yellow]")
                continue
        
        # If all models failed, raise the last exception
        logger.error("All models failed to generate image")
        if self.console:
            self.console.print(f"[bold red]❌ All models failed![/bold red]")
        raise last_exception or Exception("All models failed to generate image")
    
    async def submit_image_generation(
        self, 
        prompt: str, 
        stream: bool = True,
        conversation_id: Optional[str] = None,
        input_image: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Submit an image generation task and return a task ID immediately.
        The actual generation happens asynchronously.
        
        Args:
            prompt: The image generation prompt
            stream: Whether to use streaming response
            conversation_id: Optional conversation ID for maintaining history (auto-generated if None)
            input_image: Base64 encoded reference image (optional)
            **kwargs: Additional parameters (quality, size, format, etc.)
            
        Returns:
            Task ID that can be used to retrieve results from task_barrier()
        """
        task_id = str(uuid.uuid4())
        
        # Create the coroutine for async execution
        coro = self._async_generate_image(
            task_id=task_id,
            prompt=prompt,
            stream=stream,
            conversation_id=conversation_id,
            input_image=input_image,
            **kwargs
        )
        
        # Submit to rolling group for concurrent execution
        await self._rolling_group.submit(coro)
        
        return task_id
    
    async def task_barrier(self) -> Dict[str, Dict[str, Any]]:
        """
        Wait for all submitted tasks to complete and return their results.
        
        Returns:
            Dictionary mapping task IDs to their results
        """
        # Wait for all tasks to complete
        await self._rolling_group.barrier()
        
        # Collect and return results
        results = {}
        for task_result in self._task_results:
            results[task_result["task_id"]] = task_result["result"]
        
        # Clear results for next batch
        self._task_results.clear()
        
        return results
    
    async def _async_generate_image(
        self, 
        task_id: str,
        prompt: str, 
        stream: bool = True,
        conversation_id: Optional[str] = None,
        input_image: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Internal async method for generating images.
        Results are stored in self._task_results.
        
        Args:
            task_id: Unique task identifier
            prompt: The image generation prompt
            stream: Whether to use streaming response
            conversation_id: Optional conversation ID for maintaining history (auto-generated if None)
            input_image: Base64 encoded reference image (optional)
            **kwargs: Additional parameters (quality, size, format, etc.)
        """
        try:
            result = await self._async_generate_image_internal(
                prompt=prompt,
                stream=stream,
                conversation_id=conversation_id,
                input_image=input_image,
                **kwargs
            )
            
            # Store the result
            self._task_results.append({
                "task_id": task_id,
                "result": result,
                "error": None
            })
            
        except Exception as e:
            # Store the error
            self._task_results.append({
                "task_id": task_id,
                "result": None,
                "error": str(e)
            })
    
    async def _async_generate_image_internal(
        self, 
        prompt: str, 
        stream: bool = True,
        conversation_id: Optional[str] = None,
        input_image: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Internal async image generation method with automatic model fallback
        
        Args:
            prompt: The image generation prompt
            stream: Whether to use streaming response
            conversation_id: Optional conversation ID for maintaining history (auto-generated if None)
            input_image: Base64 encoded reference image (optional)
            **kwargs: Additional parameters (quality, size, format, etc.)
            
        Returns:
            Dictionary containing the API response, model used, and conversation info
        """
        
        # Handle conversation management with auto-generated ID
        if conversation_id is None and self.conversation_manager:
            conversation_id = self.conversation_manager.generate_conversation_id("image")
        
        conversation_info = {"conversation_id": conversation_id, "conversation_continued": False}
        
        # Build messages list starting with conversation history
        messages = []
        
        # Load conversation history if conversation_id is provided
        if conversation_id and self.conversation_manager:
            try:
                history = self.conversation_manager.load_conversation(conversation_id, "image")
                messages.extend(history)
                if history:
                    conversation_info["conversation_continued"] = True
                    if self.console:
                        self.console.print(f"[blue]📖 Loaded conversation {conversation_id} with {len(history)} messages[/blue]")
            except Exception as e:
                logger.warning(f"Failed to load conversation {conversation_id}: {e}")
                # Continue without history
        
        # Build the current message content with image generation parameters
        if input_image:
            # Use multimodal format with image and text
            content_parts = [
                {"type": "text", "text": prompt}
            ]
            
            # Add image generation parameters if provided
            if any(k in kwargs for k in ['quality', 'size', 'format', 'background', 'output_compression']):
                params = []
                if 'quality' in kwargs and kwargs['quality'] != 'auto':
                    params.append(f"quality: {kwargs['quality']}")
                if 'size' in kwargs and kwargs['size'] != 'auto':
                    params.append(f"size: {kwargs['size']}")
                if 'format' in kwargs and kwargs['format'] != 'png':
                    params.append(f"format: {kwargs['format']}")
                if 'background' in kwargs and kwargs['background'] == 'transparent':
                    params.append("background: transparent")
                if 'output_compression' in kwargs:
                    params.append(f"compression: {kwargs['output_compression']}")
                
                if params:
                    content_parts[0]["text"] += f"\n\nImage parameters: {', '.join(params)}"
            
            # Add the input image
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": input_image
                }
            })
            
            # Add current user message to the conversation
            messages.append({
                "role": "user",
                "content": content_parts
            })
        else:
            # Use text-only format
            content = prompt
            
            # Add image generation parameters if provided
            if any(k in kwargs for k in ['quality', 'size', 'format', 'background', 'output_compression']):
                params = []
                if 'quality' in kwargs and kwargs['quality'] != 'auto':
                    params.append(f"quality: {kwargs['quality']}")
                if 'size' in kwargs and kwargs['size'] != 'auto':
                    params.append(f"size: {kwargs['size']}")
                if 'format' in kwargs and kwargs['format'] != 'png':
                    params.append(f"format: {kwargs['format']}")
                if 'background' in kwargs and kwargs['background'] == 'transparent':
                    params.append("background: transparent")
                if 'output_compression' in kwargs:
                    params.append(f"compression: {kwargs['output_compression']}")
                
                if params:
                    content += f"\n\nImage parameters: {', '.join(params)}"
            
            # Add current user message to the conversation
            messages.append({
                "role": "user",
                "content": content
            })
        
        # Try models in order from lowest to highest price
        last_exception = None
        
        for model in MODEL_FALLBACK_ORDER:
            try:
                # Log to stderr for debugging (works in MCP server)
                logger.info(f"Trying model: {model}")
                
                # Also display in console if available (CLI mode)
                if self.console:
                    self.console.print(f"[dim]🤖 Trying model: {model}[/dim]")
                
                data = {
                    "model": model,
                    "stream": stream,
                    "messages": messages
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                async with aiohttp.ClientSession() as session:
                    if stream:
                        # For streaming response
                        async with session.post(
                            self.api_url, 
                            json=data, 
                            headers=headers, 
                            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                        ) as response:
                            
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"API Error: {response.status} - {error_text}")
                            
                            # Process the streaming response
                            result = await self._process_async_stream(response)
                            # Add the successful model to the result
                            result["model_used"] = model
                            result["conversation_info"] = conversation_info
                            
                            # Save conversation if conversation_id is provided
                            if conversation_id and self.conversation_manager:
                                try:
                                    # Extract assistant response from result
                                    assistant_content = ""
                                    if "choices" in result and len(result["choices"]) > 0:
                                        assistant_content = result["choices"][0].get("message", {}).get("content", "")
                                    elif "content" in result:
                                        assistant_content = result["content"]
                                    
                                    if assistant_content:
                                        # Add assistant message to conversation
                                        messages.append({
                                            "role": "assistant",
                                            "content": assistant_content
                                        })
                                        
                                        # Save updated conversation
                                        self.conversation_manager.save_conversation(conversation_id, "image", messages)
                                        
                                        if self.console:
                                            self.console.print(f"[green]💾 Conversation {conversation_id} saved[/green]")
                                except Exception as e:
                                    logger.warning(f"Failed to save conversation {conversation_id}: {e}")
                                    # Don't fail the entire operation for conversation save errors
                            
                            # Log success to stderr
                            logger.info(f"Successfully generated with model: {model}")
                            
                            # Also display in console if available (CLI mode)
                            if self.console:
                                self.console.print(f"[green]✅ Successfully generated with model: {model}[/green]")
                            return result
                    else:
                        # For non-streaming response
                        async with session.post(
                            self.api_url, 
                            json=data, 
                            headers=headers, 
                            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                        ) as response:
                            
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"API Error: {response.status} - {error_text}")
                            
                            result = await response.json()
                            
                            if "error" in result:
                                raise Exception(f"API Error: {result['error']['message']}")
                            
                            # Add the successful model to the result
                            result["model_used"] = model
                            result["conversation_info"] = conversation_info
                            
                            # Save conversation if conversation_id is provided
                            if conversation_id and self.conversation_manager:
                                try:
                                    # Extract assistant response from result
                                    assistant_content = ""
                                    if "choices" in result and len(result["choices"]) > 0:
                                        assistant_content = result["choices"][0].get("message", {}).get("content", "")
                                    elif "content" in result:
                                        assistant_content = result["content"]
                                    
                                    if assistant_content:
                                        # Add assistant message to conversation
                                        messages.append({
                                            "role": "assistant",
                                            "content": assistant_content
                                        })
                                        
                                        # Save updated conversation
                                        self.conversation_manager.save_conversation(conversation_id, "image", messages)
                                        
                                        if self.console:
                                            self.console.print(f"[green]💾 Conversation {conversation_id} saved[/green]")
                                except Exception as e:
                                    logger.warning(f"Failed to save conversation {conversation_id}: {e}")
                                    # Don't fail the entire operation for conversation save errors
                            
                            # Log success to stderr
                            logger.info(f"Successfully generated with model: {model}")
                            
                            # Also display in console if available (CLI mode)
                            if self.console:
                                self.console.print(f"[green]✅ Successfully generated with model: {model}[/green]")
                            return result
                        
            except Exception as e:
                last_exception = e
                
                # Log failure to stderr
                logger.warning(f"Model {model} failed: {e}")
                
                # Also display in console if available (CLI mode)
                if self.console:
                    self.console.print(f"[yellow]⚠️ Model {model} failed: {e}[/yellow]")
                continue
        
        # If all models failed, raise the last exception
        logger.error("All models failed to generate image")
        if self.console:
            self.console.print(f"[bold red]❌ All models failed![/bold red]")
        raise last_exception or Exception("All models failed to generate image")
    
    async def _process_async_stream(self, response) -> Dict[str, Any]:
        """Process async streaming response from Tu-zi.com API with improved progress tracking"""
        # Log queuing status to stderr
        logger.info("Starting image generation - queuing")
        
        # Check for both Chinese and English queue indicators (only if console available)
        if self.console:
            self.console.print("\n[bold cyan]🕐 Queuing / 排队中...[/bold cyan]")
        
        # Initialize progress tracking
        progress_bar = None
        progress_task = None
        current_progress = 0
        full_content = ""
        result = {}
        generation_started = False
        
        try:
            async for line in response.content:
                if not line:
                    continue
                    
                line = line.decode('utf-8').strip()
                
                # Remove 'data: ' prefix if present
                if line.startswith('data: '):
                    line = line[6:]
                
                # Skip keep-alive lines
                if line == '[DONE]':
                    break
                    
                try:
                    data = json.loads(line)
                    
                    # Extract progress information
                    if "choices" in data and len(data["choices"]) > 0:
                        message = data["choices"][0].get("delta", {})
                        content = message.get("content", "")
                        
                        if content:
                            full_content += content
                            
                            # Check for generation start indicators (Chinese and English)
                            if any(indicator in content for indicator in ["生成中", "Generating", "正在生成", "Creating"]):
                                if not generation_started:
                                    # Log generation start to stderr
                                    logger.info("Image generation started")
                                    
                                    # Display in console if available (CLI mode)
                                    if self.console:
                                        self.console.print("[bold cyan]⚡ Generating / 生成中...[/bold cyan]")
                                    generation_started = True
                                    
                                    # Initialize progress bar (only if console available)
                                    if self.console and progress_bar is None:
                                        progress_bar = Progress(
                                            SpinnerColumn(),
                                            TextColumn("[bold blue] Progress / 进度[/bold blue]"),
                                            BarColumn(bar_width=40),
                                            TaskProgressColumn(),
                                        )
                                        progress_task = progress_bar.add_task("", total=100)
                                        progress_bar.start()
                            
                            # Extract progress numbers using regex for both formats
                            # Look for patterns like "Progress 25" or "进度 25" or just numbers with dots
                            progress_matches = re.findall(r'(?:Progress|进度|完成)\s*[：:]*\s*(\d+)[%％]?|(\d+)[%％]|(\d+)\.+', content)
                            for match in progress_matches:
                                try:
                                    # Get the number from any capture group
                                    progress_num = next(p for p in match if p)
                                    if progress_num:
                                        new_progress = int(progress_num)
                                        if new_progress > current_progress and new_progress <= 100:
                                            current_progress = new_progress
                                            # Log progress to stderr
                                            logger.info(f"Generation progress: {current_progress}%")
                                            
                                            # Update progress bar if available
                                            if progress_bar and progress_task is not None:
                                                progress_bar.update(progress_task, completed=current_progress)
                                except (ValueError, IndexError):
                                    pass
                            
                            # Check for completion indicators
                            if any(indicator in content for indicator in ["生成完成", "Generation complete", "完成", "✅", "Done"]):
                                # Log completion to stderr
                                logger.info("Image generation completed")
                                
                                # Update progress bar if available
                                if progress_bar:
                                    progress_bar.update(progress_task, completed=100)
                                    progress_bar.stop()
                                
                                # Display completion in console if available
                                if self.console:
                                    self.console.print("[bold green]✅ Generation complete / 生成完成[/bold green]\n")
                                
                    # Store the last received data as the result
                    result = data
                    
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            # Log error to stderr
            logger.error(f"Error processing async stream: {e}")
            
            # Display error in console if available
            if self.console:
                self.console.print(f"[bold red]Error processing stream:[/bold red] {e}")
            
        finally:
            if progress_bar:
                progress_bar.stop()
                
        return {
            "result": result,
            "content": full_content
        }
    
    def _process_stream(self, response) -> Dict[str, Any]:
        """Process streaming response from Tu-zi.com API with improved progress tracking"""
        # Log queuing status to stderr
        logger.info("Starting image generation - queuing")
        
        # Check for both Chinese and English queue indicators (only if console available)
        if self.console:
            self.console.print("\n[bold cyan]🕐 Queuing / 排队中...[/bold cyan]")
        
        # Initialize progress tracking
        progress_bar = None
        progress_task = None
        current_progress = 0
        full_content = ""
        result = {}
        generation_started = False
        
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
                    
                    # Extract progress information
                    if "choices" in data and len(data["choices"]) > 0:
                        message = data["choices"][0].get("delta", {})
                        content = message.get("content", "")
                        
                        if content:
                            full_content += content
                            
                            # Check for generation start indicators (Chinese and English)
                            if any(indicator in content for indicator in ["生成中", "Generating", "正在生成", "Creating"]):
                                if not generation_started:
                                    # Log generation start to stderr
                                    logger.info("Image generation started")
                                    
                                    # Display in console if available (CLI mode)
                                    if self.console:
                                        self.console.print("[bold cyan]⚡ Generating / 生成中...[/bold cyan]")
                                    generation_started = True
                                    
                                    # Initialize progress bar (only if console available)
                                    if self.console and progress_bar is None:
                                        progress_bar = Progress(
                                            SpinnerColumn(),
                                            TextColumn("[bold blue] Progress / 进度[/bold blue]"),
                                            BarColumn(bar_width=40),
                                            TaskProgressColumn(),
                                        )
                                        progress_task = progress_bar.add_task("", total=100)
                                        progress_bar.start()
                            
                            # Extract progress numbers using regex for both formats
                            # Look for patterns like "Progress 25" or "进度 25" or just numbers with dots
                            progress_matches = re.findall(r'(?:Progress|进度|完成)\s*[：:]*\s*(\d+)[%％]?|(\d+)[%％]|(\d+)\.+', content)
                            for match in progress_matches:
                                try:
                                    # Get the number from any capture group
                                    progress_num = next(p for p in match if p)
                                    if progress_num:
                                        new_progress = int(progress_num)
                                        if new_progress > current_progress and new_progress <= 100:
                                            current_progress = new_progress
                                            # Log progress to stderr
                                            logger.info(f"Generation progress: {current_progress}%")
                                            
                                            # Update progress bar if available
                                            if progress_bar and progress_task is not None:
                                                progress_bar.update(progress_task, completed=current_progress)
                                except (ValueError, IndexError):
                                    pass
                            
                            # Check for completion indicators
                            if any(indicator in content for indicator in ["生成完成", "Generation complete", "完成", "✅", "Done"]):
                                # Log completion to stderr
                                logger.info("Image generation completed")
                                
                                # Update progress bar if available
                                if progress_bar:
                                    progress_bar.update(progress_task, completed=100)
                                    progress_bar.stop()
                                
                                # Display completion in console if available
                                if self.console:
                                    self.console.print("[bold green]✅ Generation complete / 生成完成[/bold green]\n")
                                
                    # Store the last received data as the result
                    result = data
                    
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            # Log error to stderr
            logger.error(f"Error processing stream: {e}")
            
            # Display error in console if available
            if self.console:
                self.console.print(f"[bold red]Error processing stream:[/bold red] {e}")
            
        finally:
            if progress_bar:
                progress_bar.stop()
                
        return {
            "result": result,
            "content": full_content
        }
    
    def extract_response_content(self, result: Dict[str, Any]) -> str:
        """Extract the content from API response"""
        try:
            if isinstance(result, dict) and "content" in result:
                return result["content"]
            elif "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                return "No content found in response"
        except Exception as e:
            # Log error to stderr
            logger.error(f"Error extracting content: {e}")
            
            # Display error in console if available
            if self.console:
                self.console.print(f"[bold red]Error extracting content:[/bold red] {e}")
            return str(result)
    
    def extract_image_urls(self, content: str) -> List[str]:
        """Extract filesystem.site image URLs from the response content"""
        # Pattern to match filesystem.site URLs
        url_pattern = r'https://filesystem\.site/cdn/(?:download/)?(\d{8})/([a-zA-Z0-9]+)\.(?:png|jpg|jpeg|webp)'
        urls = re.findall(url_pattern, content)
        
        # Convert to full download URLs and remove duplicates
        download_urls = []
        seen_filenames = set()
        
        for date, filename in urls:
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
        
        # Log extracted URLs to stderr
        logger.info(f"Extracted {len(download_urls)} image URL(s) from response")
        
        return download_urls
    
    def download_images(
        self, 
        urls: List[str], 
        output_dir: str = "images", 
        base_name: Optional[str] = None
    ) -> List[str]:
        """Download images from filesystem.site URLs"""
        if not urls:
            # Log to stderr
            logger.warning("No image URLs found in response")
            
            # Display in console if available
            if self.console:
                self.console.print("[yellow]⚠️ No image URLs found in response[/yellow]")
            return []
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        downloaded_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log download start to stderr
        logger.info(f"Starting download of {len(urls)} image(s)")
        
        # Create progress bar only if console is available
        if self.console:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Downloading images...[/bold blue]"),
                BarColumn(),
                TaskProgressColumn(),
            )
            progress.start()
            task = progress.add_task("", total=len(urls))
        else:
            progress = None
            task = None
        
        try:
            for i, url in enumerate(urls):
                try:
                    # Generate filename
                    if base_name:
                        filename = f"{base_name}_{i+1}_{timestamp}.png"
                    else:
                        filename = f"tuzi_image_{i+1}_{timestamp}.png"
                    
                    # Detect file extension from URL
                    if url.endswith('.jpg') or url.endswith('.jpeg'):
                        filename = filename.replace('.png', '.jpg')
                    elif url.endswith('.webp'):
                        filename = filename.replace('.png', '.webp')
                    
                    filepath = Path(output_dir) / filename
                    
                    # Log download attempt to stderr
                    logger.info(f"Downloading image {i+1}/{len(urls)}: {url}")
                    
                    # Download the image
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Save the image
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    downloaded_files.append(str(filepath))
                    
                    # Log success to stderr
                    logger.info(f"Successfully downloaded: {filepath}")
                    
                    # Display in console if available
                    if self.console:
                        self.console.print(f"[green]✅ Downloaded:[/green] {filepath}")
                    
                except Exception as e:
                    # Log error to stderr
                    logger.error(f"Failed to download {url}: {e}")
                    
                    # Display error in console if available
                    if self.console:
                        self.console.print(f"[red]❌ Failed to download {url}:[/red] {e}")
                
                # Update progress if available
                if progress and task is not None:
                    progress.update(task, advance=1)
        
        finally:
            # Stop progress bar if it was created
            if progress:
                progress.stop()
        
        return downloaded_files
    
    def generate_flux_image(
        self, 
        prompt: str, 
        input_image: Optional[str] = None,
        seed: Optional[int] = None,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate image using Tu-zi.com FLUX API (flux-kontext-pro model)
        
        Args:
            prompt: The image generation prompt
            input_image: Base64 encoded reference image (optional)
            seed: Reproducible generation seed (optional)
            aspect_ratio: Image dimensions ratio (default: "1:1")
            output_format: Output image format (default: "png")
            conversation_id: Optional conversation ID for maintaining history (auto-generated if None)
            
        Returns:
            Dictionary containing the API response and conversation info
        """
        
        # Handle conversation management with auto-generated ID
        if conversation_id is None and self.conversation_manager:
            conversation_id = self.conversation_manager.generate_conversation_id("flux")
        
        conversation_info = {"conversation_id": conversation_id, "conversation_continued": False}
        
        # Validate aspect ratio
        if aspect_ratio not in FLUX_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio. Must be one of: {', '.join(FLUX_ASPECT_RATIOS)}")
        
        # Validate output format
        if output_format not in FLUX_OUTPUT_FORMATS:
            raise ValueError(f"Invalid output_format. Must be one of: {', '.join(FLUX_OUTPUT_FORMATS)}")
        
        # Build messages list starting with conversation history
        messages = []
        
        # Load conversation history if conversation_id is provided
        if conversation_id and self.conversation_manager:
            try:
                history = self.conversation_manager.load_conversation(conversation_id, "flux")
                messages.extend(history)
                if history:
                    conversation_info["conversation_continued"] = True
                    if self.console:
                        self.console.print(f"[blue]📖 Loaded conversation {conversation_id} with {len(history)} messages[/blue]")
            except Exception as e:
                logger.warning(f"Failed to load conversation {conversation_id}: {e}")
        
        # Add current user message with FLUX parameters
        user_message = {
            "role": "user",
            "content": f"Generate a FLUX image with the following specifications:\n\nPrompt: {prompt}\nAspect Ratio: {aspect_ratio}\nOutput Format: {output_format}"
        }
        
        if seed is not None:
            user_message["content"] += f"\nSeed: {seed}"
        if input_image:
            user_message["content"] += f"\nReference image provided"
            
        messages.append(user_message)
        
        # Build request data
        data = {
            "model": "flux-kontext-pro",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "safety_tolerance": 6,  # Set to 6 as requested (least restrictive)
            "prompt_upsampling": True  # Set to true as requested
        }
        
        # Add optional parameters
        if input_image:
            data["input_image"] = input_image
        if seed is not None:
            data["seed"] = seed
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Log to stderr for debugging
        logger.info(f"Generating FLUX image with model: flux-kontext-pro")
        
        # Display in console if available (CLI mode)
        if self.console:
            self.console.print(f"[dim]🎨 Using FLUX model: flux-kontext-pro[/dim]")
        
        try:
            # Use the standard images/generations endpoint
            flux_url = "https://api.tu-zi.com/v1/images/generations"
            
            response = requests.post(
                flux_url, 
                json=data, 
                headers=headers, 
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"FLUX API Error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if "error" in result:
                raise Exception(f"FLUX API Error: {result['error']['message']}")
            
            result["conversation_info"] = conversation_info
            
            # Save conversation if conversation_id is provided
            if conversation_id and self.conversation_manager:
                try:
                    # For FLUX, we'll create a simple assistant response indicating successful generation
                    assistant_content = f"Successfully generated FLUX image with prompt: {prompt}"
                    if "data" in result and len(result["data"]) > 0:
                        assistant_content += f"\nGenerated {len(result['data'])} image(s)"
                    
                    # Add assistant message to conversation
                    messages.append({
                        "role": "assistant",
                        "content": assistant_content
                    })
                    
                    # Save updated conversation
                    self.conversation_manager.save_conversation(conversation_id, "flux", messages)
                    
                    if self.console:
                        self.console.print(f"[green]💾 Conversation {conversation_id} saved[/green]")
                except Exception as e:
                    logger.warning(f"Failed to save conversation {conversation_id}: {e}")
                    # Don't fail the entire operation for conversation save errors
            
            # Log success to stderr
            logger.info(f"Successfully generated FLUX image")
            
            # Display in console if available (CLI mode)
            if self.console:
                self.console.print(f"[green]✅ Successfully generated FLUX image[/green]")
            
            return result
            
        except Exception as e:
            # Log failure to stderr
            logger.error(f"FLUX model failed: {e}")
            
            # Display in console if available (CLI mode)
            if self.console:
                self.console.print(f"[red]❌ FLUX model failed: {e}[/red]")
            raise e
    
    async def submit_flux_image_generation(
        self, 
        prompt: str, 
        input_image: Optional[str] = None,
        seed: Optional[int] = None,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Submit a FLUX image generation task and return a task ID immediately.
        The actual generation happens asynchronously.
        
        Args:
            prompt: The image generation prompt
            input_image: Base64 encoded reference image (optional)
            seed: Reproducible generation seed (optional)
            aspect_ratio: Image dimensions ratio (default: "1:1")
            output_format: Output image format (default: "png")
            conversation_id: Optional conversation ID for maintaining history (auto-generated if None)
            
        Returns:
            Task ID that can be used to retrieve results from task_barrier()
        """
        task_id = str(uuid.uuid4())
        
        # Create the coroutine for async execution
        coro = self._async_generate_flux_image(
            task_id=task_id,
            prompt=prompt,
            input_image=input_image,
            seed=seed,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            conversation_id=conversation_id
        )
        
        # Submit to rolling group for concurrent execution
        await self._rolling_group.submit(coro)
        
        return task_id
    
    async def _async_generate_flux_image(
        self, 
        task_id: str,
        prompt: str, 
        input_image: Optional[str] = None,
        seed: Optional[int] = None,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        conversation_id: Optional[str] = None
    ) -> None:
        """
        Internal async method for generating FLUX images.
        Results are stored in self._task_results.
        
        Args:
            task_id: Unique task identifier
            prompt: The image generation prompt
            input_image: Base64 encoded reference image (optional)
            seed: Reproducible generation seed (optional)
            aspect_ratio: Image dimensions ratio (default: "1:1")
            output_format: Output image format (default: "png")
            conversation_id: Optional conversation ID for maintaining history (auto-generated if None)
        """
        try:
            result = await self._async_generate_flux_image_internal(
                prompt=prompt,
                input_image=input_image,
                seed=seed,
                aspect_ratio=aspect_ratio,
                output_format=output_format,
                conversation_id=conversation_id
            )
            
            # Store the result
            self._task_results.append({
                "task_id": task_id,
                "result": result,
                "error": None
            })
            
        except Exception as e:
            # Store the error
            self._task_results.append({
                "task_id": task_id,
                "result": None,
                "error": str(e)
            })
    
    async def _async_generate_flux_image_internal(
        self, 
        prompt: str, 
        input_image: Optional[str] = None,
        seed: Optional[int] = None,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Internal async FLUX image generation method
        
        Args:
            prompt: The image generation prompt
            input_image: Base64 encoded reference image (optional)
            seed: Reproducible generation seed (optional)
            aspect_ratio: Image dimensions ratio (default: "1:1")
            output_format: Output image format (default: "png")
            conversation_id: Optional conversation ID for maintaining history (auto-generated if None)
            
        Returns:
            Dictionary containing the API response and conversation info
        """
        
        # Handle conversation management with auto-generated ID
        if conversation_id is None and self.conversation_manager:
            conversation_id = self.conversation_manager.generate_conversation_id("flux")
        
        conversation_info = {"conversation_id": conversation_id, "conversation_continued": False}
        
        # Validate aspect ratio
        if aspect_ratio not in FLUX_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio. Must be one of: {', '.join(FLUX_ASPECT_RATIOS)}")
        
        # Validate output format
        if output_format not in FLUX_OUTPUT_FORMATS:
            raise ValueError(f"Invalid output_format. Must be one of: {', '.join(FLUX_OUTPUT_FORMATS)}")
        
        # Build messages list starting with conversation history
        messages = []
        
        # Load conversation history if conversation_id is provided
        if conversation_id and self.conversation_manager:
            try:
                history = self.conversation_manager.load_conversation(conversation_id, "flux")
                messages.extend(history)
                if history:
                    conversation_info["conversation_continued"] = True
                    if self.console:
                        self.console.print(f"[blue]📖 Loaded conversation {conversation_id} with {len(history)} messages[/blue]")
            except Exception as e:
                logger.warning(f"Failed to load conversation {conversation_id}: {e}")
        
        # Add current user message with FLUX parameters
        user_message = {
            "role": "user",
            "content": f"Generate a FLUX image with the following specifications:\n\nPrompt: {prompt}\nAspect Ratio: {aspect_ratio}\nOutput Format: {output_format}"
        }
        
        if seed is not None:
            user_message["content"] += f"\nSeed: {seed}"
        if input_image:
            user_message["content"] += f"\nReference image provided"
            
        messages.append(user_message)
        
        # Build request data
        data = {
            "model": "flux-kontext-pro",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "safety_tolerance": 6,  # Set to 6 as requested (least restrictive)
            "prompt_upsampling": True  # Set to true as requested
        }
        
        # Add optional parameters
        if input_image:
            data["input_image"] = input_image
        if seed is not None:
            data["seed"] = seed
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Log to stderr for debugging
        logger.info(f"Generating FLUX image with model: flux-kontext-pro")
        
        # Display in console if available (CLI mode)
        if self.console:
            self.console.print(f"[dim]🎨 Using FLUX model: flux-kontext-pro[/dim]")
        
        try:
            # Use the standard images/generations endpoint
            flux_url = "https://api.tu-zi.com/v1/images/generations"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    flux_url, 
                    json=data, 
                    headers=headers, 
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"FLUX API Error: {response.status} - {error_text}")
                    
                    result = await response.json()
                    
                    if "error" in result:
                        raise Exception(f"FLUX API Error: {result['error']['message']}")
                    
                    result["conversation_info"] = conversation_info
                    
                    # Save conversation if conversation_id is provided
                    if conversation_id and self.conversation_manager:
                        try:
                            # For FLUX, we'll create a simple assistant response indicating successful generation
                            assistant_content = f"Successfully generated FLUX image with prompt: {prompt}"
                            if "data" in result and len(result["data"]) > 0:
                                assistant_content += f"\nGenerated {len(result['data'])} image(s)"
                            
                            # Add assistant message to conversation
                            messages.append({
                                "role": "assistant",
                                "content": assistant_content
                            })
                            
                            # Save updated conversation
                            self.conversation_manager.save_conversation(conversation_id, "flux", messages)
                            
                            if self.console:
                                self.console.print(f"[green]💾 Conversation {conversation_id} saved[/green]")
                        except Exception as e:
                            logger.warning(f"Failed to save conversation {conversation_id}: {e}")
                            # Don't fail the entire operation for conversation save errors
                    
                    # Log success to stderr
                    logger.info(f"Successfully generated FLUX image")
                    
                    # Display in console if available (CLI mode)
                    if self.console:
                        self.console.print(f"[green]✅ Successfully generated FLUX image[/green]")
                    
                    return result
                    
        except Exception as e:
            # Log failure to stderr
            logger.error(f"FLUX model failed: {e}")
            
            # Display in console if available (CLI mode)
            if self.console:
                self.console.print(f"[red]❌ FLUX model failed: {e}[/red]")
            raise e
    
    async def _async_generate_image_with_path(
        self, 
        prompt: str, 
        output_path: str,
        stream: bool = True,
        conversation_id: Optional[str] = None,
        input_image: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Async image generation with automatic download to specified path
        """
        # Generate image using internal method
        result = await self._async_generate_image_internal(
            prompt=prompt,
            stream=stream,
            conversation_id=conversation_id,
            input_image=input_image,
            **kwargs
        )
        
        # Extract image URLs
        content = self.extract_response_content(result)
        image_urls = self.extract_image_urls(content)
        
        # Download to specified path if URLs found
        if image_urls:
            output_dir = os.path.dirname(output_path) or "."
            base_name = os.path.splitext(os.path.basename(output_path))[0] or "generated_image"
            
            os.makedirs(output_dir, exist_ok=True)
            downloaded_files = self.download_images(
                [image_urls[0]], 
                output_dir=output_dir, 
                base_name=base_name
            )
            
            # Copy to exact output path if different
            if downloaded_files and downloaded_files[0] != output_path:
                import shutil
                shutil.move(downloaded_files[0], output_path)
                result["downloaded_file"] = output_path
            else:
                result["downloaded_file"] = downloaded_files[0] if downloaded_files else ""
        else:
            result["downloaded_file"] = ""
            
        return result
    
    async def _async_generate_flux_image_with_path(
        self, 
        prompt: str, 
        output_path: str,
        input_image: Optional[str] = None,
        seed: Optional[int] = None,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async FLUX image generation with automatic download to specified path
        """
        # Generate image using internal method
        result = await self._async_generate_flux_image_internal(
            prompt=prompt,
            input_image=input_image,
            seed=seed,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            conversation_id=conversation_id
        )
        
        # Extract image URLs
        image_urls = self.extract_flux_image_urls(result)
        
        # Download to specified path if URLs found
        if image_urls:
            output_dir = os.path.dirname(output_path) or "."
            base_name = os.path.splitext(os.path.basename(output_path))[0] or "flux_generated_image"
            
            os.makedirs(output_dir, exist_ok=True)
            downloaded_files = self.download_images(
                [image_urls[0]], 
                output_dir=output_dir, 
                base_name=base_name
            )
            
            # Copy to exact output path if different
            if downloaded_files and downloaded_files[0] != output_path:
                import shutil
                shutil.move(downloaded_files[0], output_path)
                result["downloaded_file"] = output_path
            else:
                result["downloaded_file"] = downloaded_files[0] if downloaded_files else ""
        else:
            result["downloaded_file"] = ""
            
        return result
    
    def extract_flux_image_urls(self, result: Dict[str, Any]) -> List[str]:
        """
        Extract image URLs from FLUX API response
        
        Args:
            result: The FLUX API response dictionary
            
        Returns:
            List of image URLs
        """
        try:
            if "data" in result and isinstance(result["data"], list):
                urls = []
                for item in result["data"]:
                    if "url" in item:
                        urls.append(item["url"])
                
                # Log extracted URLs to stderr
                logger.info(f"Extracted {len(urls)} FLUX image URL(s) from response")
                
                return urls
            else:
                logger.warning("No data field found in FLUX response")
                return []
        except Exception as e:
            logger.error(f"Error extracting FLUX image URLs: {e}")
            return []


class TuZiSurvey:
    """Survey class for conducting queries using Tu-zi.com's o3-all model with web search capabilities"""
    
    def __init__(self, api_key: str, console: Optional[Console] = None, show_thinking: bool = False, conversation_manager: Optional[ConversationManager] = None):
        """
        Initialize the TuZi Survey
        
        Args:
            api_key: Tu-zi.com API key
            console: Rich console for output (optional)
            show_thinking: Whether to display the thinking process (default: False)
            conversation_manager: ConversationManager instance for handling conversation history (optional)
        """
        self.api_key = api_key
        self.api_url = "https://api.tu-zi.com/v1/chat/completions"
        self.console = console or Console()
        self.show_thinking = show_thinking
        self.conversation_manager = conversation_manager
    
    def survey(
        self, 
        prompt: str,
        stream: bool = True,
        deep: bool = False,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Conduct a survey/query using Tu-zi.com's model with web search capabilities
        
        Args:
            prompt: The natural language query/question
            stream: Whether to use streaming response
            deep: Whether to use o3-pro for deeper analysis (default: False, uses o3-all)
            conversation_id: Optional conversation ID for maintaining history (auto-generated if None)
            
        Returns:
            Dictionary containing the API response and conversation info
        """
        
        # Handle conversation management with auto-generated ID
        if conversation_id is None and self.conversation_manager:
            conversation_id = self.conversation_manager.generate_conversation_id("survey")
        
        conversation_info = {"conversation_id": conversation_id, "conversation_continued": False}
        
        # Select model based on deep parameter
        model = "o3-pro" if deep else "o3-all"
        
        # Build messages list starting with conversation history
        messages = []
        
        # Load conversation history if conversation_id is provided
        if conversation_id and self.conversation_manager:
            try:
                history = self.conversation_manager.load_conversation(conversation_id, "survey")
                messages.extend(history)
                if history:
                    conversation_info["conversation_continued"] = True
                    if self.console:
                        self.console.print(f"[blue]📖 Loaded conversation {conversation_id} with {len(history)} messages[/blue]")
            except Exception as e:
                logger.warning(f"Failed to load conversation {conversation_id}: {e}")
                # Continue without history
        
        # Add current user message to the conversation
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Log survey start to stderr
        logger.info(f"Starting survey with {model} model: {prompt[:100]}...")
        
        # Display in console if available (CLI mode)
        if self.console:
            model_display = "o3-pro (deep analysis)" if deep else "o3-all"
            self.console.print(f"[bold cyan]🔍 Surveying with {model_display} model...[/bold cyan]")
        
        data = {
            "model": model,
            "stream": stream,
            "messages": messages
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            if stream:
                # For streaming response
                with requests.post(
                    self.api_url, 
                    json=data, 
                    headers=headers, 
                    timeout=300,  # 5 minutes timeout
                    stream=True
                ) as response:
                    
                    if response.status_code != 200:
                        raise Exception(f"API Error: {response.status_code} - {response.text}")
                    
                    # Process the streaming response
                    result = self._process_survey_stream(response)
                    result["conversation_info"] = conversation_info
                    
                    # Save conversation if conversation_id is provided
                    if conversation_id and self.conversation_manager:
                        try:
                            # Extract assistant response from result
                            assistant_content = ""
                            if "content" in result:
                                assistant_content = result["content"]
                            elif "result" in result and "choices" in result["result"] and len(result["result"]["choices"]) > 0:
                                assistant_content = result["result"]["choices"][0].get("message", {}).get("content", "")
                            
                            if assistant_content:
                                # Add assistant message to conversation (filtering will be applied automatically)
                                self.conversation_manager.add_message(conversation_id, "survey", "assistant", assistant_content)
                                
                                if self.console:
                                    self.console.print(f"[green]💾 Conversation {conversation_id} saved[/green]")
                        except Exception as e:
                            logger.warning(f"Failed to save conversation {conversation_id}: {e}")
                            # Don't fail the entire operation for conversation save errors
                    
                    # Log success to stderr
                    logger.info("Survey completed successfully")
                    
                    # Display in console if available (CLI mode)
                    if self.console:
                        self.console.print(f"[green]✅ Survey completed[/green]")
                    return result
            else:
                # For non-streaming response
                response = requests.post(
                    self.api_url, 
                    json=data, 
                    headers=headers, 
                    timeout=300  # 5 minutes timeout
                )
                
                if response.status_code != 200:
                    raise Exception(f"API Error: {response.status_code} - {response.text}")
                
                result = response.json()
                
                if "error" in result:
                    raise Exception(f"API Error: {result['error']['message']}")
                
                result["conversation_info"] = conversation_info
                
                # Save conversation if conversation_id is provided
                if conversation_id and self.conversation_manager:
                    try:
                        # Extract assistant response from result
                        assistant_content = ""
                        if "choices" in result and len(result["choices"]) > 0:
                            assistant_content = result["choices"][0].get("message", {}).get("content", "")
                        elif "content" in result:
                            assistant_content = result["content"]
                        
                        if assistant_content:
                            # Add assistant message to conversation (filtering will be applied automatically)
                            self.conversation_manager.add_message(conversation_id, "survey", "assistant", assistant_content)
                            
                            if self.console:
                                self.console.print(f"[green]💾 Conversation {conversation_id} saved[/green]")
                    except Exception as e:
                        logger.warning(f"Failed to save conversation {conversation_id}: {e}")
                        # Don't fail the entire operation for conversation save errors
                
                # Log success to stderr
                logger.info("Survey completed successfully")
                
                # Display in console if available (CLI mode)
                if self.console:
                    self.console.print(f"[green]✅ Survey completed[/green]")
                return result
                
        except Exception as e:
            # Log error to stderr
            logger.error(f"Survey failed: {e}")
            
            # Display error in console if available (CLI mode)
            if self.console:
                self.console.print(f"[bold red]❌ Survey failed:[/bold red] {e}")
            raise e
    
    def _process_survey_stream(self, response) -> Dict[str, Any]:
        """Process streaming response from Tu-zi.com API for survey with time-based markdown rendering"""
        # Log processing start to stderr
        logger.info("Processing survey stream response")
        
        full_content = ""
        result = {}
        thinking_complete = False
        thinking_time_shown = False
        markdown_content = ""
        
        # For CLI mode with console, use Live rendering for markdown
        if self.console:
            if self.show_thinking:
                self.console.print("\n[bold cyan]🤔 Thinking and searching...[/bold cyan]\n")
                return self._process_with_live_markdown(response)
            else:
                self.console.print("\n[bold cyan]🤔 Thinking...[/bold cyan]")
        
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
                    
                    # Extract content
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            full_content += content
                            
                            # Display streaming content in console if available
                            if self.console:
                                # Only show busy indicators and final answer when thinking is disabled
                                import re
                                
                                # Check if we hit the thinking completion marker
                                thought_pattern = r'\*Thought for [^*]+\*'
                                match = re.search(thought_pattern, content)
                                
                                if match and not thinking_time_shown:
                                    # Show the thinking time and start showing content after
                                    thinking_time_shown = True
                                    thinking_complete = True
                                    thinking_text = match.group(0)
                                    
                                    # Clear the "Thinking..." line and show thinking time
                                    self.console.print(f"\r> {thinking_text}")
                                    
                                    # Start markdown content after thinking marker
                                    after_thinking = content[match.end():].strip()
                                    markdown_content = after_thinking  # Even if empty, start the live markdown
                                    return self._process_remaining_with_live_markdown(response, markdown_content, data)
                                elif thinking_complete:
                                    # We should not reach here as we return above
                                    pass
                                # During thinking phase, only show dots as busy indicator occasionally
                                elif len(full_content) % 50 == 0:  # Show dots every 50 characters
                                    self.console.print(".", end="")
                    
                    # Store the last received data as the result
                    result = data
                    
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            # Log error to stderr
            logger.error(f"Error processing survey stream: {e}")
            
            # Display error in console if available
            if self.console:
                self.console.print(f"\n[bold red]Error processing stream:[/bold red] {e}")
        
        # Display completion in console if available
        if self.console:
            self.console.print("\n\n[bold green]✅ Survey response complete[/bold green]\n")
                
        return {
            "result": result,
            "content": full_content
        }
    
    def _process_with_live_markdown(self, response) -> Dict[str, Any]:
        """Process stream with live markdown rendering when show_thinking is True"""
        full_content = ""
        result = {}
        
        try:
            with Live(Markdown(""), console=self.console, refresh_per_second=2) as live:
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
                        
                        # Extract content
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            
                            if content:
                                full_content += content
                                # Update live markdown display
                                try:
                                    live.update(Markdown(full_content))
                                except Exception:
                                    # Fallback to plain text if markdown fails
                                    live.update(full_content)
                        
                        # Store the last received data as the result
                        result = data
                        
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            # Log error to stderr
            logger.error(f"Error processing survey stream: {e}")
            
            # Display error in console if available
            if self.console:
                self.console.print(f"\n[bold red]Error processing stream:[/bold red] {e}")
        
        # Display completion
        if self.console:
            self.console.print("\n[bold green]✅ Survey response complete[/bold green]\n")
                
        return {
            "result": result,
            "content": full_content
        }
    
    def _process_remaining_with_live_markdown(self, response, initial_content: str, initial_result) -> Dict[str, Any]:
        """Process remaining stream with live markdown after thinking is complete"""
        full_content = initial_content
        result = initial_result
        
        try:
            # Start with empty or initial content
            display_content = initial_content if initial_content.strip() else ""
            with Live(Markdown(display_content) if display_content else "", console=self.console, refresh_per_second=2) as live:
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
                        
                        # Extract content
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            
                            if content:
                                full_content += content
                                # Update live markdown display
                                try:
                                    live.update(Markdown(full_content))
                                except Exception:
                                    # Fallback to plain text if markdown fails
                                    live.update(full_content)
                        
                        # Store the last received data as the result
                        result = data
                        
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            # Log error to stderr
            logger.error(f"Error processing survey stream: {e}")
            
            # Display error in console if available
            if self.console:
                self.console.print(f"\n[bold red]Error processing stream:[/bold red] {e}")
        
        # Display completion
        if self.console:
            self.console.print("\n[bold green]✅ Survey response complete[/bold green]\n")
                
        return {
            "result": result,
            "content": full_content
        }
    
    def extract_survey_content(self, result: Dict[str, Any]) -> str:
        """Extract the content from survey API response"""
        try:
            if isinstance(result, dict) and "content" in result:
                raw_content = result["content"]
            elif "choices" in result and len(result["choices"]) > 0:
                raw_content = result["choices"][0]["message"]["content"]
            else:
                return "No content found in response"
            
            # Parse thinking and final answer
            return self._parse_response_content(raw_content)
            
        except Exception as e:
            # Log error to stderr
            logger.error(f"Error extracting survey content: {e}")
            
            # Display error in console if available
            if self.console:
                self.console.print(f"[bold red]Error extracting content:[/bold red] {e}")
            return str(result)
    
    def _parse_response_content(self, content: str) -> str:
        """Parse response content to separate thinking from final answer"""
        if not self.show_thinking:
            import re
            
            # The actual separator pattern from o3-all responses: "*Thought for X seconds*"
            # This can be seconds, minutes and seconds (like "1m 29s"), etc.
            thought_pattern = r'\*Thought for [^*]+\*'
            
            # Split content on the thinking separator
            parts = re.split(thought_pattern, content, maxsplit=1)
            
            if len(parts) > 1:
                # Found the separator, return content after it
                final_answer = parts[1].strip()
                if final_answer:
                    return final_answer
            
            # If no separator found, return original content
            # (No fallback assumptions - only use what we know exists)
            return content
        else:
            # Return full content including thinking
            return content





def get_api_key() -> str:
    """Get API key from environment variable"""
    api_key = os.getenv("TUZI_API_KEY")
    if not api_key:
        raise ValueError("TUZI_API_KEY environment variable not set")
    return api_key


def validate_parameters(
    quality: str,
    size: str,
    format: str,
    background: str,
    compression: Optional[int] = None
) -> None:
    """Validate generation parameters"""
    if quality not in QUALITY_OPTIONS:
        raise ValueError(f"Invalid quality: {quality}. Must be one of: {', '.join(QUALITY_OPTIONS)}")
    
    if size not in SIZE_OPTIONS:
        raise ValueError(f"Invalid size: {size}. Must be one of: {', '.join(SIZE_OPTIONS)}")
    
    if format not in FORMAT_OPTIONS:
        raise ValueError(f"Invalid format: {format}. Must be one of: {', '.join(FORMAT_OPTIONS)}")
    
    if background not in BACKGROUND_OPTIONS:
        raise ValueError(f"Invalid background: {background}. Must be one of: {', '.join(BACKGROUND_OPTIONS)}")
    
    if compression is not None and (compression < 0 or compression > 100):
        raise ValueError(f"Invalid compression: {compression}. Must be between 0 and 100")
    
    # Validate background transparency only works with PNG/WebP
    if background == "transparent" and format not in ["png", "webp"]:
        raise ValueError("Transparent background only supported with PNG or WebP format")