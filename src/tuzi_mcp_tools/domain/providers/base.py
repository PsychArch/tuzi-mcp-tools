"""
Abstract base classes for image generation providers

This module defines the contract that all image generation providers must follow,
implementing the Strategy pattern for different image generation services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ImageFormat(Enum):
    """Supported image formats"""

    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"


class ImageQuality(Enum):
    """Image quality levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class BackgroundType(Enum):
    """Background types"""

    OPAQUE = "opaque"
    TRANSPARENT = "transparent"


@dataclass
class ImageGenerationRequest:
    """Request parameters for image generation"""

    prompt: str
    input_image: Optional[str] = None  # Base64 encoded reference image
    conversation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ImageGenerationResult:
    """Result from image generation"""

    success: bool
    image_urls: List[str]
    raw_response: Dict[str, Any]
    model_used: Optional[str] = None
    conversation_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    @classmethod
    def success_result(
        cls,
        image_urls: List[str],
        raw_response: Dict[str, Any],
        model_used: Optional[str] = None,
        conversation_info: Optional[Dict[str, Any]] = None,
    ) -> "ImageGenerationResult":
        """Create a successful result"""
        return cls(
            success=True,
            image_urls=image_urls,
            raw_response=raw_response,
            model_used=model_used,
            conversation_info=conversation_info,
        )

    @classmethod
    def error_result(
        cls, error_message: str, raw_response: Optional[Dict[str, Any]] = None
    ) -> "ImageGenerationResult":
        """Create an error result"""
        return cls(
            success=False,
            image_urls=[],
            raw_response=raw_response or {},
            error_message=error_message,
        )


class ImageProvider(ABC):
    """
    Abstract base class for image generation providers

    This class defines the contract that all image generation providers must implement.
    It uses the Strategy pattern to allow different providers (GPT, FLUX, etc.) to be
    used interchangeably.
    """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider"""
        pass

    @abstractmethod
    def validate_request(self, request: ImageGenerationRequest) -> None:
        """
        Validate the generation request

        Args:
            request: Image generation request to validate

        Raises:
            ValueError: If the request is invalid for this provider
        """
        pass

    @abstractmethod
    def generate_sync(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """
        Generate image synchronously

        Args:
            request: Image generation request

        Returns:
            Image generation result
        """
        pass

    @abstractmethod
    async def generate_async(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """
        Generate image asynchronously

        Args:
            request: Image generation request

        Returns:
            Image generation result
        """
        pass

    @abstractmethod
    def extract_image_urls(self, raw_response: Dict[str, Any]) -> List[str]:
        """
        Extract image URLs from provider-specific response format

        Args:
            raw_response: Raw response from the provider's API

        Returns:
            List of image URLs
        """
        pass

    def supports_streaming(self) -> bool:
        """
        Check if this provider supports streaming responses

        Returns:
            True if streaming is supported, False otherwise
        """
        return False

    def supports_conversation_history(self) -> bool:
        """
        Check if this provider supports conversation history

        Returns:
            True if conversation history is supported, False otherwise
        """
        return False

    def get_supported_formats(self) -> List[ImageFormat]:
        """
        Get list of supported image formats

        Returns:
            List of supported ImageFormat enums
        """
        return [ImageFormat.PNG, ImageFormat.JPEG, ImageFormat.WEBP]

    def get_supported_qualities(self) -> List[ImageQuality]:
        """
        Get list of supported quality levels

        Returns:
            List of supported ImageQuality enums
        """
        return [
            ImageQuality.LOW,
            ImageQuality.MEDIUM,
            ImageQuality.HIGH,
            ImageQuality.AUTO,
        ]


class StreamingImageProvider(ImageProvider):
    """
    Abstract base class for providers that support streaming responses

    This extends the base ImageProvider to add streaming-specific functionality.
    """

    def supports_streaming(self) -> bool:
        """Streaming providers always support streaming"""
        return True

    @abstractmethod
    def generate_streaming_sync(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[callable] = None,
    ) -> ImageGenerationResult:
        """
        Generate image with streaming response (synchronous)

        Args:
            request: Image generation request
            progress_callback: Optional callback for progress updates

        Returns:
            Image generation result
        """
        pass

    @abstractmethod
    async def generate_streaming_async(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[callable] = None,
    ) -> ImageGenerationResult:
        """
        Generate image with streaming response (asynchronous)

        Args:
            request: Image generation request
            progress_callback: Optional callback for progress updates

        Returns:
            Image generation result
        """
        pass


class ConversationalImageProvider(ImageProvider):
    """
    Abstract base class for providers that support conversation history

    This extends the base ImageProvider to add conversation management.
    """

    def supports_conversation_history(self) -> bool:
        """Conversational providers always support conversation history"""
        return True

    @abstractmethod
    def build_conversation_messages(
        self, request: ImageGenerationRequest, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build conversation messages including history

        Args:
            request: Current image generation request
            history: Previous conversation messages

        Returns:
            Complete list of conversation messages
        """
        pass
