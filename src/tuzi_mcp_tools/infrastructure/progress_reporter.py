"""
Progress reporting infrastructure for CLI and console output

This module provides utilities for displaying progress, status updates,
and formatted output using Rich console components.
"""

import re
import logging
from typing import Optional, Dict, Any, Callable
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.markdown import Markdown
from rich.live import Live

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Handles progress reporting and console output"""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._active_progress = None
        self._active_task = None
    
    def show_status(self, message: str, style: str = "cyan") -> None:
        """
        Show a status message
        
        Args:
            message: Status message to display
            style: Rich style for the message
        """
        if self.console:
            self.console.print(f"[{style}]{message}[/{style}]")
        logger.info(message)
    
    def show_success(self, message: str) -> None:
        """Show a success message"""
        self.show_status(f"✅ {message}", "green")
    
    def show_warning(self, message: str) -> None:
        """Show a warning message"""
        self.show_status(f"⚠️ {message}", "yellow")
    
    def show_error(self, message: str) -> None:
        """Show an error message"""
        self.show_status(f"❌ {message}", "red")
    
    def show_info(self, message: str) -> None:
        """Show an info message"""
        self.show_status(f"ℹ️ {message}", "blue")
    
    def start_progress(self, description: str, total: Optional[int] = None) -> None:
        """
        Start a progress bar
        
        Args:
            description: Description for the progress bar
            total: Total number of items (None for indeterminate)
        """
        if not self.console:
            return
            
        if self._active_progress:
            self.stop_progress()
        
        self._active_progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{description}[/bold blue]"),
            BarColumn() if total else "",
            TaskProgressColumn() if total else "",
        )
        
        self._active_task = self._active_progress.add_task("", total=total)
        self._active_progress.start()
    
    def update_progress(self, advance: int = 1, completed: Optional[int] = None) -> None:
        """
        Update the active progress bar
        
        Args:
            advance: Number of items to advance by
            completed: Set absolute completed count
        """
        if self._active_progress and self._active_task is not None:
            if completed is not None:
                self._active_progress.update(self._active_task, completed=completed)
            else:
                self._active_progress.update(self._active_task, advance=advance)
    
    def stop_progress(self) -> None:
        """Stop the active progress bar"""
        if self._active_progress:
            self._active_progress.stop()
            self._active_progress = None
            self._active_task = None


class StreamProgressTracker:
    """Tracks progress from streaming API responses"""
    
    def __init__(self, reporter: ProgressReporter):
        self.reporter = reporter
        self.current_progress = 0
        self.generation_started = False
        self.progress_active = False
    
    def process_content(self, content: str) -> None:
        """
        Process streaming content and update progress
        
        Args:
            content: New content chunk from stream
        """
        # Check for generation start indicators (Chinese and English)
        if any(indicator in content for indicator in ["生成中", "Generating", "正在生成", "Creating"]):
            if not self.generation_started:
                self.reporter.show_status("⚡ Generating / 生成中...", "cyan")
                self.generation_started = True
                
                if not self.progress_active:
                    self.reporter.start_progress("Progress / 进度", 100)
                    self.progress_active = True
        
        # Extract progress numbers using regex for both formats
        progress_matches = re.findall(
            r'(?:Progress|进度|完成)\s*[：:]*\s*(\d+)[%％]?|(\d+)[%％]|(\d+)\.+', 
            content
        )
        
        for match in progress_matches:
            try:
                # Get the number from any capture group
                progress_num = next(p for p in match if p)
                if progress_num:
                    new_progress = int(progress_num)
                    if new_progress > self.current_progress and new_progress <= 100:
                        self.current_progress = new_progress
                        logger.info(f"Generation progress: {self.current_progress}%")
                        
                        if self.progress_active:
                            self.reporter.update_progress(completed=self.current_progress)
            except (ValueError, IndexError):
                continue
        
        # Check for completion indicators
        if any(indicator in content for indicator in ["生成完成", "Generation complete", "完成", "✅", "Done"]):
            if self.progress_active:
                self.reporter.update_progress(completed=100)
                self.reporter.stop_progress()
                self.progress_active = False
            
            self.reporter.show_success("Generation complete / 生成完成")
    
    def cleanup(self) -> None:
        """Clean up progress tracking"""
        if self.progress_active:
            self.reporter.stop_progress()
            self.progress_active = False


class SurveyRenderer:
    """Handles rendering of survey responses with thinking/markdown"""
    
    def __init__(self, reporter: ProgressReporter, show_thinking: bool = False):
        self.reporter = reporter
        self.show_thinking = show_thinking
        self.thinking_complete = False
        self.thinking_time_shown = False
    
    def render_stream(self, content_stream: Callable[[], str]) -> str:
        """
        Render streaming survey response
        
        Args:
            content_stream: Function that yields content chunks
            
        Returns:
            Final rendered content
        """
        if self.show_thinking:
            return self._render_with_full_markdown(content_stream)
        else:
            return self._render_with_thinking_filter(content_stream)
    
    def _render_with_full_markdown(self, content_stream: Callable[[], str]) -> str:
        """Render with full markdown including thinking process"""
        self.reporter.show_status("🤔 Thinking and searching...", "cyan")
        
        full_content = ""
        
        if self.reporter.console:
            with Live(Markdown(""), console=self.reporter.console, refresh_per_second=2) as live:
                for chunk in content_stream():
                    full_content += chunk
                    try:
                        live.update(Markdown(full_content))
                    except Exception:
                        # Fallback to plain text if markdown fails
                        live.update(full_content)
        else:
            # No console available, just accumulate content
            for chunk in content_stream():
                full_content += chunk
        
        return full_content
    
    def _render_with_thinking_filter(self, content_stream: Callable[[], str]) -> str:
        """Render with thinking process filtered out"""
        self.reporter.show_status("🤔 Thinking...", "cyan")
        
        full_content = ""
        markdown_content = ""
        
        for chunk in content_stream():
            full_content += chunk
            
            # Check if we hit the thinking completion marker
            thought_pattern = r'\*Thought for [^*]+\*'
            match = re.search(thought_pattern, chunk)
            
            if match and not self.thinking_time_shown:
                # Show the thinking time and start showing content after
                self.thinking_time_shown = True
                self.thinking_complete = True
                thinking_text = match.group(0)
                
                # Show thinking time
                if self.reporter.console:
                    self.reporter.console.print(f"> {thinking_text}")
                
                # Start markdown content after thinking marker
                after_thinking = chunk[match.end():].strip()
                markdown_content = after_thinking
                
                # Continue with live markdown for remaining content
                return self._render_remaining_with_markdown(content_stream, markdown_content)
            
            elif self.thinking_complete:
                # Add to markdown content after thinking is complete
                markdown_content += chunk
            
            # During thinking phase, show occasional progress dots
            elif len(full_content) % 50 == 0 and self.reporter.console:
                self.reporter.console.print(".", end="")
        
        return full_content
    
    def _render_remaining_with_markdown(
        self, 
        content_stream: Callable[[], str], 
        initial_content: str
    ) -> str:
        """Render remaining content with live markdown"""
        full_content = initial_content
        
        if self.reporter.console:
            display_content = initial_content if initial_content.strip() else ""
            with Live(
                Markdown(display_content) if display_content else "", 
                console=self.reporter.console, 
                refresh_per_second=2
            ) as live:
                for chunk in content_stream():
                    full_content += chunk
                    try:
                        live.update(Markdown(full_content))
                    except Exception:
                        # Fallback to plain text if markdown fails
                        live.update(full_content)
        else:
            # No console available, just accumulate content
            for chunk in content_stream():
                full_content += chunk
        
        return full_content


class ModelFallbackTracker:
    """Tracks model fallback attempts for image generation"""
    
    def __init__(self, reporter: ProgressReporter):
        self.reporter = reporter
    
    def trying_model(self, model: str) -> None:
        """Report that we're trying a specific model"""
        self.reporter.show_status(f"🤖 Trying model: {model}", "dim")
        logger.info(f"Trying model: {model}")
    
    def model_succeeded(self, model: str) -> None:
        """Report that a model succeeded"""
        self.reporter.show_success(f"Successfully generated with model: {model}")
        logger.info(f"Successfully generated with model: {model}")
    
    def model_failed(self, model: str, error: str) -> None:
        """Report that a model failed"""
        self.reporter.show_warning(f"Model {model} failed: {error}")
        logger.warning(f"Model {model} failed: {error}")
    
    def all_models_failed(self) -> None:
        """Report that all models failed"""
        self.reporter.show_error("All models failed!")
        logger.error("All models failed to generate image")