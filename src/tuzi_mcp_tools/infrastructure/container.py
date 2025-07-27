"""
Dependency injection container for the application

This module provides a simple dependency injection container that wires up
all the components of the application according to the clean architecture.
"""

import os
import logging
from typing import Optional, Dict, Any
from rich.console import Console

from .api_client import TuZiApiClient, ApiConfig
from .file_manager import FileManager, BatchDownloader, DownloadConfig
from .progress_reporter import ProgressReporter, StreamProgressTracker, ModelFallbackTracker
from ..domain.providers.gpt_provider import GptImageProvider
from ..domain.providers.flux_provider import FluxImageProvider
from ..domain.services.conversation_service import ConversationService
from ..application.image_service import ImageGenerationService
from ..application.survey_service import SurveyService
from ..application.task_service import TaskManagementService

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Simple dependency injection container"""
    
    def __init__(self, api_key: Optional[str] = None, console: Optional[Console] = None):
        """
        Initialize the service container
        
        Args:
            api_key: Tu-zi.com API key (will try to get from environment if not provided)
            console: Rich console instance (will create one if not provided)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or self._get_api_key()
        
        # Console for CLI output
        self.console = console or Console()
        
        # Initialize components lazily
        self._api_config: Optional[ApiConfig] = None
        self._api_client: Optional[TuZiApiClient] = None
        self._file_manager: Optional[FileManager] = None
        self._batch_downloader: Optional[BatchDownloader] = None
        self._progress_reporter: Optional[ProgressReporter] = None
        self._stream_progress_tracker: Optional[StreamProgressTracker] = None
        self._model_fallback_tracker: Optional[ModelFallbackTracker] = None
        self._gpt_provider: Optional[GptImageProvider] = None
        self._flux_provider: Optional[FluxImageProvider] = None
        self._conversation_service: Optional[ConversationService] = None
        self._image_service: Optional[ImageGenerationService] = None
        self._survey_service: Optional[SurveyService] = None
        self._task_service: Optional[TaskManagementService] = None
    
    def get_api_config(self) -> ApiConfig:
        """Get API configuration"""
        if self._api_config is None:
            self._api_config = ApiConfig(api_key=self.api_key)
        return self._api_config
    
    def get_api_client(self) -> TuZiApiClient:
        """Get API client"""
        if self._api_client is None:
            self._api_client = TuZiApiClient(self.get_api_config())
        return self._api_client
    
    def get_file_manager(self) -> FileManager:
        """Get file manager"""
        if self._file_manager is None:
            download_config = DownloadConfig()
            self._file_manager = FileManager(download_config)
        return self._file_manager
    
    def get_batch_downloader(self) -> BatchDownloader:
        """Get batch downloader"""
        if self._batch_downloader is None:
            self._batch_downloader = BatchDownloader(self.get_file_manager())
        return self._batch_downloader
    
    def get_progress_reporter(self) -> ProgressReporter:
        """Get progress reporter"""
        if self._progress_reporter is None:
            self._progress_reporter = ProgressReporter(self.console)
        return self._progress_reporter
    
    def get_stream_progress_tracker(self) -> StreamProgressTracker:
        """Get stream progress tracker"""
        if self._stream_progress_tracker is None:
            self._stream_progress_tracker = StreamProgressTracker(self.get_progress_reporter())
        return self._stream_progress_tracker
    
    def get_model_fallback_tracker(self) -> ModelFallbackTracker:
        """Get model fallback tracker"""
        if self._model_fallback_tracker is None:
            self._model_fallback_tracker = ModelFallbackTracker(self.get_progress_reporter())
        return self._model_fallback_tracker
    
    def get_gpt_provider(self) -> GptImageProvider:
        """Get GPT image provider"""
        if self._gpt_provider is None:
            self._gpt_provider = GptImageProvider(
                api_client=self.get_api_client(),
                progress_tracker=self.get_stream_progress_tracker(),
                fallback_tracker=self.get_model_fallback_tracker()
            )
        return self._gpt_provider
    
    def get_flux_provider(self) -> FluxImageProvider:
        """Get FLUX image provider"""
        if self._flux_provider is None:
            self._flux_provider = FluxImageProvider(
                api_client=self.get_api_client(),
                progress_reporter=self.get_progress_reporter()
            )
        return self._flux_provider
    
    def get_conversation_service(self, storage_mode: str = "memory") -> ConversationService:
        """Get conversation service"""
        if self._conversation_service is None:
            conversation_dir = None
            if storage_mode == "file":
                from pathlib import Path
                conversation_dir = Path.cwd()
            
            self._conversation_service = ConversationService(
                storage_mode=storage_mode,
                conversation_dir=conversation_dir
            )
        return self._conversation_service
    
    def get_image_service(self) -> ImageGenerationService:
        """Get image generation service"""
        if self._image_service is None:
            self._image_service = ImageGenerationService(
                gpt_provider=self.get_gpt_provider(),
                flux_provider=self.get_flux_provider(),
                file_manager=self.get_file_manager(),
                batch_downloader=self.get_batch_downloader(),
                progress_reporter=self.get_progress_reporter(),
                conversation_service=self.get_conversation_service()
            )
        return self._image_service
    
    def get_survey_service(self) -> SurveyService:
        """Get survey service"""
        if self._survey_service is None:
            self._survey_service = SurveyService(
                api_client=self.get_api_client(),
                progress_reporter=self.get_progress_reporter(),
                conversation_service=self.get_conversation_service()
            )
        return self._survey_service
    
    def get_task_service(self) -> TaskManagementService:
        """Get task management service"""
        if self._task_service is None:
            self._task_service = TaskManagementService(
                image_service=self.get_image_service(),
                conversation_service=self.get_conversation_service()
            )
        return self._task_service
    
    
    def _get_api_key(self) -> str:
        """Get API key from environment variable"""
        api_key = os.getenv("TUZI_API_KEY")
        if not api_key:
            raise ValueError("TUZI_API_KEY environment variable not set")
        return api_key


class ContainerFactory:
    """Factory for creating service containers with different configurations"""
    
    @staticmethod
    def create_cli_container() -> ServiceContainer:
        """Create container for CLI usage with Rich console"""
        console = Console()
        return ServiceContainer(console=console)
    
    @staticmethod
    def create_mcp_container() -> ServiceContainer:
        """Create container for MCP server usage without console output"""
        # For MCP server, we don't want Rich console output
        return ServiceContainer(console=None)
    
    @staticmethod
    def create_test_container(api_key: str = "test_key") -> ServiceContainer:
        """Create container for testing with mock dependencies"""
        return ServiceContainer(api_key=api_key, console=None)


# Global container instances for convenience
_cli_container: Optional[ServiceContainer] = None
_mcp_container: Optional[ServiceContainer] = None


def get_cli_container() -> ServiceContainer:
    """Get or create the global CLI container"""
    global _cli_container
    if _cli_container is None:
        _cli_container = ContainerFactory.create_cli_container()
    return _cli_container


def get_mcp_container() -> ServiceContainer:
    """Get or create the global MCP container"""
    global _mcp_container
    if _mcp_container is None:
        _mcp_container = ContainerFactory.create_mcp_container()
    return _mcp_container


def reset_containers() -> None:
    """Reset global containers (useful for testing)"""
    global _cli_container, _mcp_container
    _cli_container = None
    _mcp_container = None