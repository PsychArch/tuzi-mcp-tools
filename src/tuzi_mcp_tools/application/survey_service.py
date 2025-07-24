"""
Application service for survey operations

This service handles survey/query operations using the Tu-zi.com o3-all and o3-pro models,
with conversation management and thinking process filtering.
"""

import logging
from typing import Optional, Dict, Any

from ..domain.entities import Survey, ConversationType
from ..domain.services.conversation_service import ConversationService
from ..infrastructure.api_client import TuZiApiClient, ApiConfig, StreamProcessor, validate_api_response
from ..infrastructure.progress_reporter import ProgressReporter, SurveyRenderer

logger = logging.getLogger(__name__)


class SurveyService:
    """Service for conducting surveys and queries"""
    
    def __init__(
        self,
        api_client: TuZiApiClient,
        progress_reporter: Optional[ProgressReporter] = None,
        conversation_service: Optional[ConversationService] = None
    ):
        self.api_client = api_client
        self.progress_reporter = progress_reporter
        self.conversation_service = conversation_service
    
    def conduct_survey(
        self,
        query: str,
        deep: bool = False,
        show_thinking: bool = False,
        stream: bool = True,
        conversation_id: Optional[str] = None
    ) -> Survey:
        """
        Conduct a survey/query operation
        
        Args:
            query: The natural language query/question
            deep: Whether to use o3-pro for deeper analysis (default: o3-all)
            show_thinking: Whether to include thinking process in response
            stream: Whether to use streaming response
            conversation_id: Optional conversation ID for maintaining history
            
        Returns:
            Survey entity with response
            
        Raises:
            Exception: If survey fails
        """
        # Handle conversation management with auto-generated ID
        if conversation_id is None and self.conversation_service:
            conversation_id = self.conversation_service.generate_conversation_id(ConversationType.SURVEY)
        
        # Select model based on deep parameter
        model = "o3-pro" if deep else "o3-all"
        
        # Build messages including conversation history
        messages = self._build_messages(query, conversation_id)
        
        # Log survey start
        logger.info(f"Starting survey with {model} model: {query[:100]}...")
        
        # Show progress
        if self.progress_reporter:
            model_display = "o3-pro (deep analysis)" if deep else "o3-all"
            self.progress_reporter.show_status(f"🔍 Surveying with {model_display} model...", "cyan")
        
        try:
            # Create survey entity
            survey = Survey(
                query=query,
                model_used=model,
                deep_analysis=deep,
                show_thinking=show_thinking,
                conversation_id=conversation_id
            )
            
            # Make API request
            data = {
                "model": model,
                "stream": stream,
                "messages": messages
            }
            
            if stream:
                response = self.api_client.post_sync("chat/completions", data, stream=True)
                result = self._process_streaming_response(response, show_thinking)
                
                # Extract content
                content = result.get("content", "")
                survey.set_response(content, self._extract_thinking_time(content))
                
            else:
                response = self.api_client.post_sync("chat/completions", data)
                result = response.json()
                validate_api_response(result)
                
                # Extract content from non-streaming response
                content = self._extract_content_from_result(result)
                survey.set_response(content, self._extract_thinking_time(content))
            
            # Save conversation if conversation service available
            if conversation_id and self.conversation_service:
                self._save_conversation(conversation_id, query, survey.get_filtered_response())
            
            # Show success
            if self.progress_reporter:
                self.progress_reporter.show_success("Survey completed")
            
            logger.info("Survey completed successfully")
            
            return survey
            
        except Exception as e:
            error_msg = f"Survey failed: {e}"
            logger.error(error_msg)
            
            if self.progress_reporter:
                self.progress_reporter.show_error(error_msg)
            
            raise Exception(error_msg)
    
    def _build_messages(self, query: str, conversation_id: Optional[str]) -> list:
        """Build messages list including conversation history"""
        messages = []
        
        # Load conversation history if available
        if conversation_id and self.conversation_service:
            try:
                history = self.conversation_service.get_conversation_messages(conversation_id, ConversationType.SURVEY)
                messages.extend(history)
                
                if history and self.progress_reporter:
                    self.progress_reporter.show_info(f"📖 Loaded conversation {conversation_id} with {len(history)} messages")
                    
            except Exception as e:
                logger.warning(f"Failed to load conversation {conversation_id}: {e}")
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
    
    def _process_streaming_response(self, response, show_thinking: bool) -> Dict[str, Any]:
        """Process streaming survey response with optional thinking display"""
        if not self.progress_reporter:
            # Fallback to basic stream processing if no progress reporter
            return StreamProcessor.process_sync_stream(response)
        
        # Use survey renderer for rich display
        renderer = SurveyRenderer(self.progress_reporter, show_thinking)
        
        full_content = ""
        result = {}
        
        def content_generator():
            nonlocal full_content, result
            
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
                    import json
                    data = json.loads(line)
                    
                    # Extract and yield content
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            yield content
                    
                    # Store result
                    result = data
                    
                except json.JSONDecodeError:
                    continue
        
        # Use renderer to process stream
        full_content = renderer.render_stream(content_generator)
        
        return {
            "result": result,
            "content": full_content
        }
    
    def _extract_content_from_result(self, result: Dict[str, Any]) -> str:
        """Extract content from non-streaming API result"""
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("message", {}).get("content", "")
        elif "content" in result:
            return result["content"]
        else:
            return "No content found in response"
    
    def _extract_thinking_time(self, content: str) -> Optional[str]:
        """Extract thinking time from response content"""
        import re
        # Look for patterns like "*Thought for 30 seconds*" or "*Thought for 1m 29s*"
        match = re.search(r'\*Thought for ([^*]+)\*', content)
        return match.group(1) if match else None
    
    def _save_conversation(self, conversation_id: str, query: str, response: str) -> None:
        """Save conversation with query and response"""
        try:
            # Add user message (query)
            self.conversation_service.add_message_to_conversation(
                conversation_id, ConversationType.SURVEY, "user", query
            )
            
            # Add assistant message (response) - filtering will be applied automatically
            self.conversation_service.add_message_to_conversation(
                conversation_id, ConversationType.SURVEY, "assistant", response
            )
            
            if self.progress_reporter:
                self.progress_reporter.show_success(f"💾 Conversation {conversation_id} saved")
                
        except Exception as e:
            logger.warning(f"Failed to save conversation {conversation_id}: {e}")
            # Don't fail the entire operation for conversation save errors