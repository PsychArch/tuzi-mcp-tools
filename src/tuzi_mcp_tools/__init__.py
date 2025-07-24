"""
Tuzi MCP Tools - CLI and MCP tool interfaces for Tu-zi.com API

This package provides both command-line interface (CLI) and Model Context Protocol (MCP)
server implementations for generating images using the Tu-zi.com API.
"""

# Import from new architecture for backward compatibility
from .infrastructure.container import ServiceContainer
from .application.image_service import ImageGenerationService
from .application.survey_service import SurveyService

# Create compatibility classes that wrap the new services
class TuZiImageGenerator:
    def __init__(self, api_key: str = None):
        self.container = ServiceContainer(api_key=api_key)
        self.image_service = self.container.get_image_service()
    
    def generate_gpt_image(self, prompt: str, **kwargs):
        return self.image_service.generate_gpt_image(prompt, **kwargs)
    
    def generate_flux_image(self, prompt: str, **kwargs):
        return self.image_service.generate_flux_image(prompt, **kwargs)

class TuZiSurvey:
    def __init__(self, api_key: str = None):
        self.container = ServiceContainer(api_key=api_key)
        self.survey_service = self.container.get_survey_service()
    
    def conduct_survey(self, query: str, **kwargs):
        return self.survey_service.conduct_survey(query, **kwargs)

__all__ = ["TuZiImageGenerator", "TuZiSurvey"] 