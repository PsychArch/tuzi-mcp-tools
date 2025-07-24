"""
Domain service for conversation management

This service handles conversation persistence, circular buffering,
and conversation-related business logic.
"""

import json
import threading
import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..entities import Conversation, ConversationMessage, ConversationType

logger = logging.getLogger(__name__)


class ConversationService:
    """Domain service for managing conversations with automatic ID generation and circular buffer"""
    
    def __init__(self, storage_mode: str = "memory", conversation_dir: Optional[Path] = None):
        """
        Initialize the conversation service
        
        Args:
            storage_mode: "memory" for in-memory storage (MCP), "file" for file storage (CLI)
            conversation_dir: Directory for file storage (defaults to current directory)
        """
        self.storage_mode = storage_mode
        self.conversation_dir = conversation_dir or Path.cwd()
        
        # Get circular buffer depth from environment variable (default: 10)
        import os
        self.max_conversations = int(os.getenv("TUZI_CONVERSATION_BUFFER_SIZE", "10"))
        
        # Use OrderedDict for LRU-style circular buffer in memory mode
        self.conversations = OrderedDict()  # conversation_key -> Conversation
        
        # Counter for generating sequential conversation IDs
        self._id_counter = 1
        self._id_lock = threading.Lock()
        
        logger.info(f"ConversationService initialized: storage_mode={storage_mode}, max_conversations={self.max_conversations}")
    
    def generate_conversation_id(self, conversation_type: ConversationType) -> str:
        """
        Generate a unique conversation ID automatically
        
        Args:
            conversation_type: Type of conversation
            
        Returns:
            Generated conversation ID
        """
        with self._id_lock:
            # Generate ID with format: {type}_{counter}
            conversation_id = f"{conversation_type.value}_{self._id_counter}"
            self._id_counter += 1
            logger.debug(f"Generated conversation ID: {conversation_id}")
            return conversation_id
    
    def create_conversation(
        self, 
        conversation_id: str, 
        conversation_type: ConversationType
    ) -> Conversation:
        """
        Create a new conversation
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_type: Type of conversation
            
        Returns:
            New Conversation entity
        """
        self._validate_conversation_id(conversation_id)
        
        conversation = Conversation(
            conversation_id=conversation_id,
            conversation_type=conversation_type
        )
        
        logger.info(f"Created new conversation: {conversation_id} ({conversation_type.value})")
        return conversation
    
    def load_conversation(
        self, 
        conversation_id: str, 
        conversation_type: ConversationType
    ) -> Optional[Conversation]:
        """
        Load conversation from storage
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_type: Type of conversation
            
        Returns:
            Conversation entity or None if not found
        """
        self._validate_conversation_id(conversation_id)
        
        conversation_key = f"{conversation_type.value}:{conversation_id}"
        
        if self.storage_mode == "memory":
            # Move to end (mark as recently accessed) if exists
            if conversation_key in self.conversations:
                conversation = self.conversations[conversation_key]
                self.conversations.move_to_end(conversation_key)
                logger.debug(f"Loaded conversation from memory: {conversation_id}")
                return conversation
            return None
        
        elif self.storage_mode == "file":
            file_path = self._get_conversation_file_path(conversation_id, conversation_type.value)
            
            if not file_path.exists():
                return None
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversation = Conversation.from_dict(data)
                    logger.debug(f"Loaded conversation from file: {conversation_id}")
                    return conversation
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load conversation {conversation_id}: {e}")
                return None
        
        return None
    
    def save_conversation(self, conversation: Conversation) -> None:
        """
        Save conversation to storage with circular buffer management
        
        Args:
            conversation: Conversation entity to save
        """
        self._validate_conversation_id(conversation.conversation_id)
        
        conversation_key = f"{conversation.conversation_type.value}:{conversation.conversation_id}"
        
        if self.storage_mode == "memory":
            # Add/update conversation
            self.conversations[conversation_key] = conversation
            # Move to end (mark as most recently used)
            self.conversations.move_to_end(conversation_key)
            
            # Implement circular buffer: remove oldest if over limit
            while len(self.conversations) > self.max_conversations:
                oldest_key = next(iter(self.conversations))
                logger.info(f"Removing oldest conversation from buffer: {oldest_key}")
                del self.conversations[oldest_key]
            
            logger.debug(f"Saved conversation to memory: {conversation.conversation_id}")
        
        elif self.storage_mode == "file":
            file_path = self._get_conversation_file_path(
                conversation.conversation_id, 
                conversation.conversation_type.value
            )
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                conversation_data = conversation.to_dict()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"Saved conversation to file: {conversation.conversation_id}")
            except IOError as e:
                logger.error(f"Failed to save conversation {conversation.conversation_id}: {e}")
                raise
    
    def add_message_to_conversation(
        self, 
        conversation_id: str, 
        conversation_type: ConversationType, 
        role: str, 
        content: str
    ) -> None:
        """
        Add a message to an existing conversation or create new one
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_type: Type of conversation
            role: Message role ("user" or "assistant")
            content: Message content
        """
        # Load existing conversation or create new one
        conversation = self.load_conversation(conversation_id, conversation_type)
        if conversation is None:
            conversation = self.create_conversation(conversation_id, conversation_type)
        
        # Filter content for survey conversations
        if conversation_type == ConversationType.SURVEY and role == "assistant":
            content = self._filter_thinking_content(content)
        
        # Add message
        conversation.add_message(role, content)
        
        # Save conversation
        self.save_conversation(conversation)
        
        logger.debug(f"Added {role} message to conversation {conversation_id}")
    
    def get_conversation_messages(
        self, 
        conversation_id: str, 
        conversation_type: ConversationType
    ) -> List[Dict[str, Any]]:
        """
        Get conversation messages in API format
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_type: Type of conversation
            
        Returns:
            List of message dictionaries in API format
        """
        conversation = self.load_conversation(conversation_id, conversation_type)
        if conversation is None:
            return []
        
        # Convert to API format
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in conversation.messages
        ]
    
    def get_conversation_summary(
        self, 
        conversation_id: str, 
        conversation_type: ConversationType
    ) -> Dict[str, Any]:
        """
        Get conversation summary information
        
        Args:
            conversation_id: Unique conversation identifier
            conversation_type: Type of conversation
            
        Returns:
            Dictionary with conversation summary
        """
        conversation = self.load_conversation(conversation_id, conversation_type)
        
        if conversation is None:
            return {
                "exists": False,
                "message_count": 0,
                "created_at": None,
                "updated_at": None,
                "last_user_message": None
            }
        
        return {
            "exists": True,
            "message_count": conversation.get_message_count(),
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "last_user_message": conversation.get_last_user_message()
        }
    
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
    
    def _validate_conversation_id(self, conversation_id: str) -> None:
        """Validate conversation ID format"""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', conversation_id):
            raise ValueError(f"Invalid conversation_id format: {conversation_id}")
    
    def _get_conversation_file_path(self, conversation_id: str, conversation_type: str) -> Path:
        """Get the file path for a conversation"""
        filename = f"tuzi-{conversation_type}-{conversation_id}.json"
        return self.conversation_dir / filename
    
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
        return content