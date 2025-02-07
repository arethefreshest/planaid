from pydantic_settings import BaseSettings
from extract_thinker import Extractor, LLM
from .llm.extractor import FieldExtractor
import os
from pathlib import Path
import litellm

# Get the root directory (where .env is located)
ROOT_DIR = Path(__file__).parent.parent.parent  # This goes from app/config.py to root

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str = "https://kartai-openai.openai.azure.com"
    OPENAI_API_VERSION: str = "2024-08-01-preview"
    OPENAI_DEPLOYMENT_NAME: str = "gpt-4o-mini"
    OPENAI_API_TYPE: str = "azure"
    
    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = 'utf-8'

settings = Settings()

# Configure litellm
litellm.api_key = settings.OPENAI_API_KEY
litellm.api_base = settings.OPENAI_API_BASE
litellm.api_version = settings.OPENAI_API_VERSION
litellm.azure = True

# The model name needs to be in Azure format
model_name = f"azure/{settings.OPENAI_DEPLOYMENT_NAME}"

# Initialize LLM with Azure configuration
llm = LLM(
    model=model_name,
    token_limit=4000
)

# Initialize the field extractor
extractor = FieldExtractor(model_name) 