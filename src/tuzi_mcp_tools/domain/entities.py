"""
Domain entities for the image generation and survey system

This module contains the core business entities that represent the main
concepts in the domain, such as Images, Conversations, and Surveys.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ConversationType(Enum):
    """Types of conversations supported"""

    IMAGE = "image"
    FLUX = "flux"
    SURVEY = "survey"


class TaskStatus(Enum):
    """Status of async tasks"""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ConversationMessage:
    """A single message in a conversation"""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary"""
        timestamp = (
            datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now()
        )
        return cls(role=data["role"], content=data["content"], timestamp=timestamp)


@dataclass
class Conversation:
    """A conversation entity representing a chat session"""

    conversation_id: str
    conversation_type: ConversationType
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    task_counter: int = field(default=0)  # Counter for generating task IDs
    pending_task_ids: List[str] = field(default_factory=list)  # Track pending tasks

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation"""
        message = ConversationMessage(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_message_count(self) -> int:
        """Get the total number of messages"""
        return len(self.messages)

    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message"""
        for message in reversed(self.messages):
            if message.role == "user":
                content = message.content
                return content[:100] + "..." if len(content) > 100 else content
        return None

    def generate_next_task_id(self) -> str:
        """Generate next task ID in sequence"""
        self.task_counter += 1
        return f"{self.conversation_id}_task_{self.task_counter}"

    def add_pending_task(self, task_id: str) -> None:
        """Add a task ID to pending tasks list"""
        if task_id not in self.pending_task_ids:
            self.pending_task_ids.append(task_id)
            self.updated_at = datetime.now()

    def remove_pending_task(self, task_id: str) -> None:
        """Remove a task ID from pending tasks list"""
        if task_id in self.pending_task_ids:
            self.pending_task_ids.remove(task_id)
            self.updated_at = datetime.now()

    def has_pending_tasks(self) -> bool:
        """Check if conversation has any pending tasks"""
        return len(self.pending_task_ids) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "conversation_id": self.conversation_id,
            "conversation_type": self.conversation_type.value,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "task_counter": self.task_counter,
            "pending_task_ids": self.pending_task_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create from dictionary"""
        return cls(
            conversation_id=data["conversation_id"],
            conversation_type=ConversationType(data["conversation_type"]),
            messages=[
                ConversationMessage.from_dict(msg) for msg in data.get("messages", [])
            ],
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                data.get("updated_at", datetime.now().isoformat())
            ),
            task_counter=data.get("task_counter", 0),
            pending_task_ids=data.get("pending_task_ids", []),
        )


@dataclass
class GeneratedImage:
    """Represents a generated image"""

    image_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    image_urls: List[str] = field(default_factory=list)
    local_paths: List[str] = field(default_factory=list)
    model_used: Optional[str] = None
    provider_name: str = ""
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    conversation_id: Optional[str] = None

    def add_local_path(self, path: str) -> None:
        """Add a local file path for the image"""
        if path not in self.local_paths:
            self.local_paths.append(path)

    def get_primary_url(self) -> Optional[str]:
        """Get the primary image URL"""
        return self.image_urls[0] if self.image_urls else None

    def get_primary_path(self) -> Optional[str]:
        """Get the primary local file path"""
        return self.local_paths[0] if self.local_paths else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "image_id": self.image_id,
            "prompt": self.prompt,
            "image_urls": self.image_urls,
            "local_paths": self.local_paths,
            "model_used": self.model_used,
            "provider_name": self.provider_name,
            "generation_parameters": self.generation_parameters,
            "created_at": self.created_at.isoformat(),
            "conversation_id": self.conversation_id,
        }


@dataclass
class Survey:
    """Represents a survey/query operation"""

    survey_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    response: str = ""
    model_used: str = "o3-all"
    deep_analysis: bool = False
    show_thinking: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    conversation_id: Optional[str] = None
    thinking_time: Optional[str] = None

    def set_response(self, response: str, thinking_time: Optional[str] = None) -> None:
        """Set the survey response"""
        self.response = response
        self.thinking_time = thinking_time

    def get_filtered_response(self) -> str:
        """Get response with thinking filtered out if show_thinking is False"""
        if self.show_thinking:
            return self.response

        # Filter out thinking content
        import re

        thought_pattern = r"\*Thought for [^*]+\*"
        parts = re.split(thought_pattern, self.response, maxsplit=1)

        if len(parts) > 1:
            final_answer = parts[1].strip()
            return final_answer if final_answer else self.response

        return self.response

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "survey_id": self.survey_id,
            "query": self.query,
            "response": self.response,
            "model_used": self.model_used,
            "deep_analysis": self.deep_analysis,
            "show_thinking": self.show_thinking,
            "created_at": self.created_at.isoformat(),
            "conversation_id": self.conversation_id,
            "thinking_time": self.thinking_time,
        }


@dataclass
class AsyncTask:
    """Represents an asynchronous task"""

    task_id: str  # Format: {conversation_id}_task_{task_sequence}
    conversation_id: str  # Required parent conversation
    task_sequence: int  # Sequential number within conversation
    task_type: str  # "gpt_image", "flux_image", "survey"
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def mark_executing(self) -> None:
        """Mark task as executing"""
        self.status = TaskStatus.EXECUTING

    def mark_completed(self, result: Dict[str, Any]) -> None:
        """Mark task as completed with result"""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed with error"""
        self.status = TaskStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now()

    def is_completed(self) -> bool:
        """Check if task is completed (success or failure)"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "task_sequence": self.task_sequence,
            "task_type": self.task_type,
            "status": self.status.value,
            "result": self.result,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


@dataclass
class ImageGenerationSession:
    """Represents a complete image generation session"""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provider_name: str = ""
    generated_images: List[GeneratedImage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    conversation_id: Optional[str] = None

    def add_image(self, image: GeneratedImage) -> None:
        """Add a generated image to the session"""
        self.generated_images.append(image)

    def get_image_count(self) -> int:
        """Get the total number of generated images"""
        return len(self.generated_images)

    def get_total_urls(self) -> int:
        """Get the total number of image URLs across all images"""
        return sum(len(img.image_urls) for img in self.generated_images)

    def get_all_local_paths(self) -> List[str]:
        """Get all local file paths from all images"""
        paths = []
        for image in self.generated_images:
            paths.extend(image.local_paths)
        return paths
