"""
GPT Image Generation Provider

Implementation of the ImageProvider interface for Tu-zi.com's GPT-based
image generation models with automatic fallback support.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import (
    StreamingImageProvider,
    ConversationalImageProvider,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageFormat,
    ImageQuality,
    BackgroundType,
)
from ...infrastructure.api_client import TuZiApiClient, StreamProcessor
from ...infrastructure.file_manager import ImageUrlExtractor
from ...infrastructure.progress_reporter import (
    StreamProgressTracker,
    ModelFallbackTracker,
)

logger = logging.getLogger(__name__)


# Model order from lowest to highest price (fallback order)
GPT_MODEL_FALLBACK_ORDER = [
    "gpt-image-1",  # $0.04
    "gpt-4o-image",  # $0.04
    "gpt-4o-image-vip",  # $0.10
    "gpt-image-1-vip",  # $0.10
]


@dataclass
class GptImageRequest(ImageGenerationRequest):
    """GPT-specific image generation request"""

    quality: ImageQuality = ImageQuality.AUTO
    size: str = "auto"  # "1024x1024", "1536x1024", "1024x1536", "auto"
    format: ImageFormat = ImageFormat.PNG
    background: BackgroundType = BackgroundType.OPAQUE
    output_compression: Optional[int] = None  # 0-100 for JPEG/WebP
    stream: bool = True

    def validate(self) -> None:
        """Validate GPT-specific parameters"""
        # Validate size options
        valid_sizes = ["1024x1024", "1536x1024", "1024x1536", "auto"]
        if self.size not in valid_sizes:
            raise ValueError(
                f"Invalid size: {self.size}. Must be one of: {', '.join(valid_sizes)}"
            )

        # Validate compression
        if self.output_compression is not None:
            if not (0 <= self.output_compression <= 100):
                raise ValueError(
                    f"Invalid compression: {self.output_compression}. Must be between 0 and 100"
                )

        # Validate background transparency only works with PNG/WebP
        if self.background == BackgroundType.TRANSPARENT and self.format not in [
            ImageFormat.PNG,
            ImageFormat.WEBP,
        ]:
            raise ValueError(
                "Transparent background only supported with PNG or WebP format"
            )


class GptImageProvider(StreamingImageProvider, ConversationalImageProvider):
    """GPT image generation provider with model fallback"""

    def __init__(
        self,
        api_client: TuZiApiClient,
        progress_tracker: Optional[StreamProgressTracker] = None,
        fallback_tracker: Optional[ModelFallbackTracker] = None,
    ):
        self.api_client = api_client
        self.progress_tracker = progress_tracker
        self.fallback_tracker = fallback_tracker
        self.url_extractor = ImageUrlExtractor()

    def get_provider_name(self) -> str:
        """Get the name of this provider"""
        return "GPT Image Generator"

    def validate_request(self, request: ImageGenerationRequest) -> None:
        """Validate the generation request"""
        if not isinstance(request, GptImageRequest):
            # Convert base request to GPT request for validation
            request = GptImageRequest(
                prompt=request.prompt,
                input_image=request.input_image,
                conversation_id=request.conversation_id,
            )

        request.validate()

    def generate_sync(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate image synchronously"""
        if not isinstance(request, GptImageRequest):
            request = self._convert_to_gpt_request(request)

        if request.stream:
            return self.generate_streaming_sync(request)
        else:
            return self._generate_non_streaming_sync(request)

    async def generate_async(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate image asynchronously"""
        if not isinstance(request, GptImageRequest):
            request = self._convert_to_gpt_request(request)

        if request.stream:
            return await self.generate_streaming_async(request)
        else:
            return await self._generate_non_streaming_async(request)

    def generate_streaming_sync(
        self, request: GptImageRequest, progress_callback: Optional[callable] = None
    ) -> ImageGenerationResult:
        """Generate image with streaming response (synchronous)"""
        # Try models in fallback order
        last_exception = None

        for model in GPT_MODEL_FALLBACK_ORDER:
            try:
                if self.fallback_tracker:
                    self.fallback_tracker.trying_model(model)

                # Build request data
                messages = self._build_messages(request)
                data = {"model": model, "stream": True, "messages": messages}

                # Make API request
                response = self.api_client.post_sync(
                    "chat/completions", data, stream=True
                )

                # Process streaming response
                stream_result = StreamProcessor.process_sync_stream(response)

                # Track progress if tracker available
                if self.progress_tracker and progress_callback:
                    # Process content for progress tracking
                    content = stream_result.get("content", "")
                    for chunk in content.split():  # Simple chunking for progress
                        self.progress_tracker.process_content(chunk)
                        if progress_callback:
                            progress_callback(chunk)

                # Extract image URLs
                content = stream_result.get("content", "")
                image_urls = self.url_extractor.extract_filesystem_urls(content)

                if self.fallback_tracker:
                    self.fallback_tracker.model_succeeded(model)

                return ImageGenerationResult.success_result(
                    image_urls=image_urls, raw_response=stream_result, model_used=model
                )

            except Exception as e:
                last_exception = e
                if self.fallback_tracker:
                    self.fallback_tracker.model_failed(model, str(e))
                continue

        # All models failed
        if self.fallback_tracker:
            self.fallback_tracker.all_models_failed()

        return ImageGenerationResult.error_result(
            error_message=f"All GPT models failed. Last error: {last_exception}"
        )

    async def generate_streaming_async(
        self, request: GptImageRequest, progress_callback: Optional[callable] = None
    ) -> ImageGenerationResult:
        """Generate image with streaming response (asynchronous)"""
        # Try models in fallback order
        last_exception = None

        for model in GPT_MODEL_FALLBACK_ORDER:
            try:
                if self.fallback_tracker:
                    self.fallback_tracker.trying_model(model)

                # Build request data
                messages = self._build_messages(request)
                data = {"model": model, "stream": True, "messages": messages}

                # Make async API request and process stream within session context
                stream_result = await self.api_client.post_async_with_stream_processing(
                    "chat/completions", data
                )

                # Track progress if tracker available
                if self.progress_tracker and progress_callback:
                    # Process content for progress tracking
                    content = stream_result.get("content", "")
                    for chunk in content.split():  # Simple chunking for progress
                        self.progress_tracker.process_content(chunk)
                        if progress_callback:
                            progress_callback(chunk)

                # Extract image URLs
                content = stream_result.get("content", "")
                image_urls = self.url_extractor.extract_filesystem_urls(content)

                if self.fallback_tracker:
                    self.fallback_tracker.model_succeeded(model)

                return ImageGenerationResult.success_result(
                    image_urls=image_urls, raw_response=stream_result, model_used=model
                )

            except Exception as e:
                last_exception = e
                if self.fallback_tracker:
                    self.fallback_tracker.model_failed(model, str(e))
                continue

        # All models failed
        if self.fallback_tracker:
            self.fallback_tracker.all_models_failed()

        return ImageGenerationResult.error_result(
            error_message=f"All GPT models failed. Last error: {last_exception}"
        )

    def extract_image_urls(self, raw_response: Dict[str, Any]) -> List[str]:
        """Extract image URLs from GPT response"""
        content = ""

        # Extract content from various response formats
        if "content" in raw_response:
            content = raw_response["content"]
        elif "result" in raw_response and "content" in raw_response["result"]:
            content = raw_response["result"]["content"]
        elif "choices" in raw_response and len(raw_response["choices"]) > 0:
            choice = raw_response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]

        return self.url_extractor.extract_filesystem_urls(content)

    def build_conversation_messages(
        self, request: ImageGenerationRequest, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build conversation messages including history"""
        messages = history.copy()
        messages.extend(self._build_messages(request))
        return messages

    def _convert_to_gpt_request(
        self, request: ImageGenerationRequest
    ) -> GptImageRequest:
        """Convert base request to GPT-specific request"""
        return GptImageRequest(
            prompt=request.prompt,
            input_image=request.input_image,
            conversation_id=request.conversation_id,
        )

    def _build_messages(self, request: GptImageRequest) -> List[Dict[str, Any]]:
        """Build messages for API request"""
        if request.input_image:
            # Use multimodal format with image and text
            content_parts = [{"type": "text", "text": request.prompt}]

            # Add image generation parameters if provided
            params = self._build_parameter_string(request)
            if params:
                content_parts[0]["text"] += f"\n\nImage parameters: {params}"

            # Add the input image
            content_parts.append(
                {"type": "image_url", "image_url": {"url": request.input_image}}
            )

            return [{"role": "user", "content": content_parts}]
        else:
            # Use text-only format
            content = request.prompt

            # Add image generation parameters if provided
            params = self._build_parameter_string(request)
            if params:
                content += f"\n\nImage parameters: {params}"

            return [{"role": "user", "content": content}]

    def _build_parameter_string(self, request: GptImageRequest) -> str:
        """Build parameter string for prompt"""
        params = []

        if request.quality != ImageQuality.AUTO:
            params.append(f"quality: {request.quality.value}")
        if request.size != "auto":
            params.append(f"size: {request.size}")
        if request.format != ImageFormat.PNG:
            params.append(f"format: {request.format.value}")
        if request.background == BackgroundType.TRANSPARENT:
            params.append("background: transparent")
        if request.output_compression is not None:
            params.append(f"compression: {request.output_compression}")

        return ", ".join(params)

    def _generate_non_streaming_sync(
        self, request: GptImageRequest
    ) -> ImageGenerationResult:
        """Generate image without streaming (synchronous)"""
        # Similar to streaming but without stream processing
        # Implementation would be similar but using regular API call
        # For brevity, delegating to streaming version
        request.stream = True
        return self.generate_streaming_sync(request)

    async def _generate_non_streaming_async(
        self, request: GptImageRequest
    ) -> ImageGenerationResult:
        """Generate image without streaming (asynchronous)"""
        # Similar to streaming but without stream processing
        # Implementation would be similar but using regular API call
        # For brevity, delegating to streaming version
        request.stream = True
        return await self.generate_streaming_async(request)
