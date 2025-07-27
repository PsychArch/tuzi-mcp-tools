"""
Application service for image generation

This service orchestrates image generation operations across different providers,
handling conversation management, file downloads, and task coordination.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

from ..domain.entities import GeneratedImage
from ..domain.providers.gpt_provider import GptImageProvider, GptImageRequest
from ..domain.providers.flux_provider import FluxImageProvider, FluxImageRequest
from ..domain.services.conversation_service import ConversationService
from ..infrastructure.file_manager import BatchDownloader, FileManager
from ..infrastructure.progress_reporter import ProgressReporter

logger = logging.getLogger(__name__)


class ImageGenerationService:
    """Service for orchestrating image generation operations"""

    def __init__(
        self,
        gpt_provider: GptImageProvider,
        flux_provider: FluxImageProvider,
        file_manager: FileManager,
        batch_downloader: BatchDownloader,
        progress_reporter: Optional[ProgressReporter] = None,
        conversation_service: Optional[ConversationService] = None,
    ):
        self.gpt_provider = gpt_provider
        self.flux_provider = flux_provider
        self.file_manager = file_manager
        self.batch_downloader = batch_downloader
        self.progress_reporter = progress_reporter
        self.conversation_service = conversation_service

    def generate_gpt_image(
        self,
        prompt: str,
        quality: str = "auto",
        size: str = "auto",
        format: str = "png",
        background: str = "opaque",
        output_compression: Optional[int] = None,
        input_image: Optional[str] = None,
        conversation_id: Optional[str] = None,
        stream: bool = True,
    ) -> GeneratedImage:
        """
        Generate image using GPT provider

        Args:
            prompt: Image generation prompt
            quality: Image quality ("auto", "low", "medium", "high")
            size: Image size ("auto", "1024x1024", "1536x1024", "1024x1536")
            format: Output format ("png", "jpeg", "webp")
            background: Background type ("opaque", "transparent")
            output_compression: Compression level for JPEG/WebP (0-100)
            input_image: Base64 encoded reference image
            conversation_id: Optional conversation ID
            stream: Whether to use streaming response

        Returns:
            GeneratedImage entity

        Raises:
            ValueError: If parameters are invalid
            Exception: If generation fails
        """
        # Create GPT-specific request
        request = GptImageRequest(
            prompt=prompt,
            input_image=input_image,
            conversation_id=conversation_id,
            quality=self._parse_quality(quality),
            size=size,
            format=self._parse_format(format),
            background=self._parse_background(background),
            output_compression=output_compression,
            stream=stream,
        )

        # Generate image
        result = self.gpt_provider.generate_sync(request)

        if not result.success:
            raise Exception(f"Image generation failed: {result.error_message}")

        # Create domain entity
        image = GeneratedImage(
            prompt=prompt,
            image_urls=result.image_urls,
            model_used=result.model_used,
            provider_name=self.gpt_provider.get_provider_name(),
            generation_parameters={
                "quality": quality,
                "size": size,
                "format": format,
                "background": background,
                "output_compression": output_compression,
                "stream": stream,
            },
            conversation_id=conversation_id,
        )

        return image

    def generate_flux_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        seed: Optional[int] = None,
        input_image: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> GeneratedImage:
        """
        Generate image using FLUX provider

        Args:
            prompt: Image generation prompt
            aspect_ratio: Image aspect ratio ("1:1", "16:9", "9:16", etc.)
            output_format: Output format ("png", "jpg", "jpeg", "webp")
            seed: Optional seed for reproducible generation
            input_image: Base64 encoded reference image
            conversation_id: Optional conversation ID

        Returns:
            GeneratedImage entity

        Raises:
            ValueError: If parameters are invalid
            Exception: If generation fails
        """
        # Create FLUX-specific request
        request = FluxImageRequest(
            prompt=prompt,
            input_image=input_image,
            conversation_id=conversation_id,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            seed=seed,
        )

        # Generate image
        result = self.flux_provider.generate_sync(request)

        if not result.success:
            raise Exception(f"FLUX image generation failed: {result.error_message}")

        # Create domain entity
        image = GeneratedImage(
            prompt=prompt,
            image_urls=result.image_urls,
            model_used=result.model_used,
            provider_name=self.flux_provider.get_provider_name(),
            generation_parameters={
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "seed": seed,
            },
            conversation_id=conversation_id,
        )

        return image

    def download_images(
        self,
        image: GeneratedImage,
        output_dir: str = "images",
        base_name: Optional[str] = None,
    ) -> List[str]:
        """
        Download images to local filesystem

        Args:
            image: GeneratedImage entity with URLs to download
            output_dir: Directory to save images
            base_name: Base name for generated filenames

        Returns:
            List of downloaded file paths
        """
        if not image.image_urls:
            if self.progress_reporter:
                self.progress_reporter.show_warning("No image URLs found for download")
            return []

        # Use prompt as base name if not provided
        if not base_name:
            # Clean prompt for filename
            base_name = self._clean_filename(image.prompt[:50])

        # Download images
        downloaded_paths = self.batch_downloader.download_images(
            urls=image.image_urls, output_dir=output_dir, base_name=base_name
        )

        # Update image entity with local paths
        for path in downloaded_paths:
            image.add_local_path(path)

        return downloaded_paths

    async def generate_gpt_image_async(
        self,
        prompt: str,
        output_path: str,
        quality: str = "auto",
        size: str = "auto",
        format: str = "png",
        background: str = "opaque",
        output_compression: Optional[int] = None,
        input_image: Optional[str] = None,
        conversation_id: Optional[str] = None,
        stream: bool = True,
    ) -> GeneratedImage:
        """
        Generate GPT image asynchronously with automatic download to specified path
        """
        # Create GPT-specific request
        request = GptImageRequest(
            prompt=prompt,
            input_image=input_image,
            conversation_id=conversation_id,
            quality=self._parse_quality(quality),
            size=size,
            format=self._parse_format(format),
            background=self._parse_background(background),
            output_compression=output_compression,
            stream=stream,
        )

        # Generate image asynchronously
        result = await self.gpt_provider.generate_async(request)

        if not result.success:
            raise Exception(f"Async image generation failed: {result.error_message}")

        # Create domain entity
        image = GeneratedImage(
            prompt=prompt,
            image_urls=result.image_urls,
            model_used=result.model_used,
            provider_name=self.gpt_provider.get_provider_name(),
            generation_parameters={
                "quality": quality,
                "size": size,
                "format": format,
                "background": background,
                "output_compression": output_compression,
                "stream": stream,
            },
            conversation_id=conversation_id,
        )

        # Download to specified path if URLs available
        if image.image_urls and output_path:
            await self._download_to_specific_path(image, output_path)

        return image

    async def generate_flux_image_async(
        self,
        prompt: str,
        output_path: str,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        seed: Optional[int] = None,
        input_image: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> GeneratedImage:
        """
        Generate FLUX image asynchronously with automatic download to specified path
        """
        # Create FLUX-specific request
        request = FluxImageRequest(
            prompt=prompt,
            input_image=input_image,
            conversation_id=conversation_id,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            seed=seed,
        )

        # Generate image asynchronously
        result = await self.flux_provider.generate_async(request)

        if not result.success:
            raise Exception(
                f"Async FLUX image generation failed: {result.error_message}"
            )

        # Create domain entity
        image = GeneratedImage(
            prompt=prompt,
            image_urls=result.image_urls,
            model_used=result.model_used,
            provider_name=self.flux_provider.get_provider_name(),
            generation_parameters={
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "seed": seed,
            },
            conversation_id=conversation_id,
        )

        # Download to specified path if URLs available
        if image.image_urls and output_path:
            await self._download_to_specific_path(image, output_path)

        return image

    def _parse_quality(self, quality: str):
        """Parse quality string to enum"""
        from ..domain.providers.base import ImageQuality

        quality_map = {
            "auto": ImageQuality.AUTO,
            "low": ImageQuality.LOW,
            "medium": ImageQuality.MEDIUM,
            "high": ImageQuality.HIGH,
        }
        return quality_map.get(quality, ImageQuality.AUTO)

    def _parse_format(self, format: str):
        """Parse format string to enum"""
        from ..domain.providers.base import ImageFormat

        format_map = {
            "png": ImageFormat.PNG,
            "jpeg": ImageFormat.JPEG,
            "jpg": ImageFormat.JPG,
            "webp": ImageFormat.WEBP,
        }
        return format_map.get(format, ImageFormat.PNG)

    def _parse_background(self, background: str):
        """Parse background string to enum"""
        from ..domain.providers.base import BackgroundType

        background_map = {
            "opaque": BackgroundType.OPAQUE,
            "transparent": BackgroundType.TRANSPARENT,
        }
        return background_map.get(background, BackgroundType.OPAQUE)

    def _clean_filename(self, text: str) -> str:
        """Clean text for use as filename"""
        import re

        # Remove or replace invalid filename characters
        cleaned = re.sub(r'[<>:"/\\|?*]', "_", text)
        # Remove extra whitespace and replace with underscores
        cleaned = re.sub(r"\s+", "_", cleaned.strip())
        return cleaned

    async def _download_to_specific_path(
        self, image: GeneratedImage, output_path: str
    ) -> None:
        """Download image to specific path"""
        if not image.image_urls:
            return

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) or "."
        base_name = (
            os.path.splitext(os.path.basename(output_path))[0] or "generated_image"
        )

        self.file_manager.ensure_directory(output_dir)

        # Download first image
        temp_paths = self.batch_downloader.download_images(
            [image.image_urls[0]], output_dir=output_dir, base_name=base_name
        )

        # Move to exact output path if different
        if temp_paths and temp_paths[0] != output_path:
            source_path = Path(temp_paths[0])
            dest_path = Path(output_path)

            if self.file_manager.move_file(source_path, dest_path):
                image.add_local_path(output_path)
            else:
                image.add_local_path(temp_paths[0])
        elif temp_paths:
            image.add_local_path(temp_paths[0])
