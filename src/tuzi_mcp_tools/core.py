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
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
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

# Configuration options
QUALITY_OPTIONS = ["low", "medium", "high", "auto"]
SIZE_OPTIONS = ["1024x1024", "1536x1024", "1024x1536", "auto"]
FORMAT_OPTIONS = ["png", "jpeg", "webp"]
BACKGROUND_OPTIONS = ["transparent", "opaque"]


class TuZiImageGenerator:
    """Main class for generating images using Tu-zi.com API"""
    
    def __init__(self, api_key: str, console: Optional[Console] = None):
        """
        Initialize the TuZi Image Generator
        
        Args:
            api_key: Tu-zi.com API key
            console: Rich console for output (optional)
        """
        self.api_key = api_key
        self.api_url = "https://api.tu-zi.com/v1/chat/completions"
        self.console = console or Console()
    
    def generate_image(
        self, 
        prompt: str, 
        stream: bool = True, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image using Tu-zi.com API with automatic model fallback
        
        Args:
            prompt: The image generation prompt
            stream: Whether to use streaming response
            **kwargs: Additional parameters (quality, size, format, etc.)
            
        Returns:
            Dictionary containing the API response and the model used
        """
        
        # Build the message content with image generation parameters
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
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
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


class TuZiSurvey:
    """Survey class for conducting queries using Tu-zi.com's o3-all model with web search capabilities"""
    
    def __init__(self, api_key: str, console: Optional[Console] = None, show_thinking: bool = False):
        """
        Initialize the TuZi Survey
        
        Args:
            api_key: Tu-zi.com API key
            console: Rich console for output (optional)
            show_thinking: Whether to display the thinking process (default: False)
        """
        self.api_key = api_key
        self.api_url = "https://api.tu-zi.com/v1/chat/completions"
        self.console = console or Console()
        self.show_thinking = show_thinking
    
    def survey(
        self, 
        prompt: str,
        stream: bool = True
    ) -> Dict[str, Any]:
        """
        Conduct a survey/query using Tu-zi.com's o3-all model with web search capabilities
        
        Args:
            prompt: The natural language query/question
            stream: Whether to use streaming response
            
        Returns:
            Dictionary containing the API response
        """
        
        # Log survey start to stderr
        logger.info(f"Starting survey with o3-all model: {prompt[:100]}...")
        
        # Display in console if available (CLI mode)
        if self.console:
            self.console.print(f"[bold cyan]🔍 Surveying with o3-all model...[/bold cyan]")
        
        data = {
            "model": "o3-all",
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
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