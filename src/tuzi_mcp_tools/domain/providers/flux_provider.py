"""
FLUX Image Generation Provider

Implementation of the ImageProvider interface for Tu-zi.com's FLUX-based
image generation using the flux-kontext-pro model.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import (
    ImageProvider, ConversationalImageProvider,
    ImageGenerationRequest, ImageGenerationResult,
    ImageFormat
)
from ...infrastructure.api_client import TuZiApiClient, ApiConfig, validate_api_response
from ...infrastructure.file_manager import ImageUrlExtractor
from ...infrastructure.progress_reporter import ProgressReporter

logger = logging.getLogger(__name__)


# FLUX-specific configuration options
FLUX_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"]
FLUX_OUTPUT_FORMATS = ["png", "jpg", "jpeg", "webp"]


@dataclass
class FluxImageRequest(ImageGenerationRequest):
    """FLUX-specific image generation request"""
    aspect_ratio: str = "1:1"
    output_format: str = "png"
    seed: Optional[int] = None
    safety_tolerance: int = 6  # Set to 6 as requested (least restrictive)
    prompt_upsampling: bool = True  # Set to true as requested
    
    def validate(self) -> None:
        """Validate FLUX-specific parameters"""
        # Validate aspect ratio
        if self.aspect_ratio not in FLUX_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio: {self.aspect_ratio}. Must be one of: {', '.join(FLUX_ASPECT_RATIOS)}")
        
        # Validate output format
        if self.output_format not in FLUX_OUTPUT_FORMATS:
            raise ValueError(f"Invalid output_format: {self.output_format}. Must be one of: {', '.join(FLUX_OUTPUT_FORMATS)}")
        
        # Validate seed if provided
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("Seed must be an integer")
        
        # Validate safety tolerance
        if not (0 <= self.safety_tolerance <= 6):
            raise ValueError("Safety tolerance must be between 0 and 6")


class FluxImageProvider(ConversationalImageProvider):
    """FLUX image generation provider"""
    
    def __init__(
        self, 
        api_client: TuZiApiClient,
        progress_reporter: Optional[ProgressReporter] = None
    ):
        self.api_client = api_client
        self.progress_reporter = progress_reporter
        self.url_extractor = ImageUrlExtractor()
    
    def get_provider_name(self) -> str:
        """Get the name of this provider"""
        return "FLUX Image Generator"
    
    def validate_request(self, request: ImageGenerationRequest) -> None:
        """Validate the generation request"""
        if not isinstance(request, FluxImageRequest):
            # Convert base request to FLUX request for validation
            request = FluxImageRequest(
                prompt=request.prompt,
                input_image=request.input_image,
                conversation_id=request.conversation_id
            )
        
        request.validate()
    
    def generate_sync(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate image synchronously"""
        if not isinstance(request, FluxImageRequest):
            request = self._convert_to_flux_request(request)
        
        try:
            if self.progress_reporter:
                self.progress_reporter.show_status("🎨 Using FLUX model: flux-kontext-pro", "dim")
            
            logger.info("Generating FLUX image with model: flux-kontext-pro")
            
            # Build request data
            data = self._build_request_data(request)
            
            # Make API request to images/generations endpoint
            response = self.api_client.post_sync("images/generations", data)
            result = response.json()
            
            # Validate response
            validate_api_response(result)
            
            # Extract image URLs
            image_urls = self.extract_image_urls(result)
            
            if self.progress_reporter:
                self.progress_reporter.show_success("Successfully generated FLUX image")
            
            logger.info("Successfully generated FLUX image")
            
            return ImageGenerationResult.success_result(
                image_urls=image_urls,
                raw_response=result,
                model_used="flux-kontext-pro"
            )
            
        except Exception as e:
            error_msg = f"FLUX model failed: {e}"
            
            if self.progress_reporter:
                self.progress_reporter.show_error(error_msg)
            
            logger.error(error_msg)
            
            return ImageGenerationResult.error_result(error_message=error_msg)
    
    async def generate_async(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate image asynchronously"""
        if not isinstance(request, FluxImageRequest):
            request = self._convert_to_flux_request(request)
        
        try:
            if self.progress_reporter:
                self.progress_reporter.show_status("🎨 Using FLUX model: flux-kontext-pro", "dim")
            
            logger.info("Generating FLUX image with model: flux-kontext-pro")
            
            # Build request data
            data = self._build_request_data(request)
            
            # Make async API request to images/generations endpoint
            response = await self.api_client.post_async("images/generations", data)
            
            async with response:
                result = await response.json()
                
                # Validate response
                validate_api_response(result)
                
                # Extract image URLs
                image_urls = self.extract_image_urls(result)
                
                if self.progress_reporter:
                    self.progress_reporter.show_success("Successfully generated FLUX image")
                
                logger.info("Successfully generated FLUX image")
                
                return ImageGenerationResult.success_result(
                    image_urls=image_urls,
                    raw_response=result,
                    model_used="flux-kontext-pro"
                )
            
        except Exception as e:
            error_msg = f"FLUX model failed: {e}"
            
            if self.progress_reporter:
                self.progress_reporter.show_error(error_msg)
            
            logger.error(error_msg)
            
            return ImageGenerationResult.error_result(error_message=error_msg)
    
    def extract_image_urls(self, raw_response: Dict[str, Any]) -> List[str]:
        """Extract image URLs from FLUX response"""
        return self.url_extractor.extract_flux_urls(raw_response)
    
    def build_conversation_messages(
        self, 
        request: ImageGenerationRequest,
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build conversation messages including history"""
        messages = history.copy()
        
        if not isinstance(request, FluxImageRequest):
            request = self._convert_to_flux_request(request)
        
        # Add current user message with FLUX parameters
        user_message = {
            "role": "user",
            "content": f"Generate a FLUX image with the following specifications:\n\nPrompt: {request.prompt}\nAspect Ratio: {request.aspect_ratio}\nOutput Format: {request.output_format}"
        }
        
        if request.seed is not None:
            user_message["content"] += f"\nSeed: {request.seed}"
        if request.input_image:
            user_message["content"] += f"\nReference image provided"
        
        messages.append(user_message)
        return messages
    
    def get_supported_formats(self) -> List[ImageFormat]:
        """Get list of supported image formats"""
        return [
            ImageFormat.PNG,
            ImageFormat.JPG,
            ImageFormat.JPEG,
            ImageFormat.WEBP
        ]
    
    def get_supported_aspect_ratios(self) -> List[str]:
        """Get list of supported aspect ratios"""
        return FLUX_ASPECT_RATIOS.copy()
    
    def _convert_to_flux_request(self, request: ImageGenerationRequest) -> FluxImageRequest:
        """Convert base request to FLUX-specific request"""
        return FluxImageRequest(
            prompt=request.prompt,
            input_image=request.input_image,
            conversation_id=request.conversation_id
        )
    
    def _build_request_data(self, request: FluxImageRequest) -> Dict[str, Any]:
        """Build request data for FLUX API"""
        data = {
            "model": "flux-kontext-pro",
            "prompt": request.prompt,
            "aspect_ratio": request.aspect_ratio,
            "output_format": request.output_format,
            "safety_tolerance": request.safety_tolerance,
            "prompt_upsampling": request.prompt_upsampling
        }
        
        # Add optional parameters
        if request.input_image:
            data["input_image"] = request.input_image
        if request.seed is not None:
            data["seed"] = request.seed
        
        return data