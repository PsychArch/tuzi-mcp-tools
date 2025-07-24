"""
Application service for async task management

This service handles the submit/barrier pattern for asynchronous image generation,
managing task lifecycle and result collection.
"""

import uuid
import asyncio
import logging
from typing import Dict, Any, Optional, List
from collections import OrderedDict
from datetime import datetime

from ..domain.entities import AsyncTask, TaskStatus, GeneratedImage
from .image_service import ImageGenerationService

logger = logging.getLogger(__name__)


class TaskManagementService:
    """Service for managing asynchronous tasks with submit/barrier pattern"""
    
    def __init__(
        self,
        image_service: ImageGenerationService,
        max_completed_tasks: int = 100
    ):
        self.image_service = image_service
        self.max_completed_tasks = max_completed_tasks
        self._lock = asyncio.Lock()
        
        # Task storage with LRU behavior
        self.executing_tasks: Dict[str, AsyncTask] = {}  # task_id -> AsyncTask
        self.asyncio_tasks: Dict[str, asyncio.Task] = {}  # task_id -> asyncio.Task
        self.completed_tasks: OrderedDict[str, AsyncTask] = OrderedDict()  # task_id -> AsyncTask
        
        logger.info(f"TaskManagementService initialized with max_completed_tasks={max_completed_tasks}")
    
    async def submit_gpt_image_task(
        self,
        prompt: str,
        output_path: str,
        quality: str = "auto",
        size: str = "auto",
        format: str = "png",
        background: str = "opaque",
        output_compression: Optional[int] = None,
        input_image: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Submit a GPT image generation task for async execution
        
        Args:
            prompt: Image generation prompt
            output_path: Path where to save the generated image
            quality: Image quality setting
            size: Image size setting
            format: Output format
            background: Background type
            output_compression: Compression level
            input_image: Base64 encoded reference image
            conversation_id: Optional conversation ID
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        
        # Create task entity
        task = AsyncTask(
            task_id=task_id,
            task_type="gpt_image"
        )
        
        # Create coroutine for execution
        coro = self._execute_gpt_image_task(
            task=task,
            prompt=prompt,
            output_path=output_path,
            quality=quality,
            size=size,
            format=format,
            background=background,
            output_compression=output_compression,
            input_image=input_image,
            conversation_id=conversation_id
        )
        
        # Submit for execution
        async with self._lock:
            task.mark_executing()
            self.executing_tasks[task_id] = task
            
            # Create and start asyncio task
            asyncio_task = asyncio.create_task(coro)
            self.asyncio_tasks[task_id] = asyncio_task
            
            logger.info(f"GPT image task {task_id} submitted and started execution")
        
        return task_id
    
    async def submit_flux_image_task(
        self,
        prompt: str,
        output_path: str,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        seed: Optional[int] = None,
        input_image: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Submit a FLUX image generation task for async execution
        
        Args:
            prompt: Image generation prompt
            output_path: Path where to save the generated image
            aspect_ratio: Image aspect ratio
            output_format: Output format
            seed: Optional seed for reproducible generation
            input_image: Base64 encoded reference image
            conversation_id: Optional conversation ID
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        
        # Create task entity
        task = AsyncTask(
            task_id=task_id,
            task_type="flux_image"
        )
        
        # Create coroutine for execution
        coro = self._execute_flux_image_task(
            task=task,
            prompt=prompt,
            output_path=output_path,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            seed=seed,
            input_image=input_image,
            conversation_id=conversation_id
        )
        
        # Submit for execution
        async with self._lock:
            task.mark_executing()
            self.executing_tasks[task_id] = task
            
            # Create and start asyncio task
            asyncio_task = asyncio.create_task(coro)
            self.asyncio_tasks[task_id] = asyncio_task
            
            logger.info(f"FLUX image task {task_id} submitted and started execution")
        
        return task_id
    
    async def wait_for_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Wait for all executing tasks to complete and return results (barrier)
        
        Returns:
            Dictionary mapping task_ids to their results/status
        """
        # Get snapshot of currently executing tasks
        async with self._lock:
            executing_tasks_snapshot = dict(self.asyncio_tasks)
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
            results = {}
            completed_count = 0
            failed_count = 0
            
            # Process all completed tasks
            for task_id, task in self.completed_tasks.items():
                if task.status == TaskStatus.COMPLETED:
                    results[task_id] = {
                        "success": True,
                        "result": task.result,
                        "error": None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None
                    }
                    completed_count += 1
                elif task.status == TaskStatus.FAILED:
                    results[task_id] = {
                        "success": False,
                        "result": None,
                        "error": task.error_message,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None
                    }
                    failed_count += 1
            
            # Clear completed tasks after collecting (barrier cleans up)
            self.completed_tasks.clear()
            
            logger.info(f"Barrier returning {len(results)} task results (completed: {completed_count}, failed: {failed_count})")
            
            # Return summary with individual results
            return {
                "completed": completed_count,
                "failed": failed_count,
                "total": completed_count + failed_count,
                "results": results
            }
    
    async def get_task_status(self) -> Dict[str, Any]:
        """Get current task manager status"""
        async with self._lock:
            return {
                "executing_tasks": len(self.executing_tasks),
                "completed_tasks": len(self.completed_tasks),
                "executing_task_ids": list(self.executing_tasks.keys()),
                "completed_task_ids": list(self.completed_tasks.keys())
            }
    
    async def _execute_gpt_image_task(
        self,
        task: AsyncTask,
        prompt: str,
        output_path: str,
        **kwargs
    ) -> None:
        """Execute GPT image generation task"""
        try:
            # Generate image using service
            image = await self.image_service.generate_gpt_image_async(
                prompt=prompt,
                output_path=output_path,
                **kwargs
            )
            
            # Create result
            result = {
                "image_id": image.image_id,
                "prompt": image.prompt,
                "image_urls": image.image_urls,
                "local_paths": image.local_paths,
                "model_used": image.model_used,
                "provider_name": image.provider_name,
                "output_path": output_path
            }
            
            # Mark task as completed
            async with self._lock:
                task.mark_completed(result)
                self._move_to_completed(task)
                
        except Exception as e:
            # Mark task as failed
            async with self._lock:
                task.mark_failed(str(e))
                self._move_to_completed(task)
    
    async def _execute_flux_image_task(
        self,
        task: AsyncTask,
        prompt: str,
        output_path: str,
        **kwargs
    ) -> None:
        """Execute FLUX image generation task"""
        try:
            # Generate image using service
            image = await self.image_service.generate_flux_image_async(
                prompt=prompt,
                output_path=output_path,
                **kwargs
            )
            
            # Create result
            result = {
                "image_id": image.image_id,
                "prompt": image.prompt,
                "image_urls": image.image_urls,
                "local_paths": image.local_paths,
                "model_used": image.model_used,
                "provider_name": image.provider_name,
                "output_path": output_path
            }
            
            # Mark task as completed
            async with self._lock:
                task.mark_completed(result)
                self._move_to_completed(task)
                
        except Exception as e:
            # Mark task as failed
            async with self._lock:
                task.mark_failed(str(e))
                self._move_to_completed(task)
    
    def _move_to_completed(self, task: AsyncTask) -> None:
        """Move task from executing to completed (must be called with lock held)"""
        task_id = task.task_id
        
        # Remove from executing
        if task_id in self.executing_tasks:
            del self.executing_tasks[task_id]
        if task_id in self.asyncio_tasks:
            del self.asyncio_tasks[task_id]
        
        # Add to completed with LRU management
        self.completed_tasks[task_id] = task
        
        # LRU cleanup - remove oldest completed tasks if over limit
        while len(self.completed_tasks) > self.max_completed_tasks:
            oldest_task_id = next(iter(self.completed_tasks))
            logger.info(f"LRU cleanup: removing oldest completed task {oldest_task_id}")
            del self.completed_tasks[oldest_task_id]
        
        logger.info(f"Task {task_id} moved to completed with status: {task.status.value}")