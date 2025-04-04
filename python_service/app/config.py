from pydantic_settings import BaseSettings
from extract_thinker import Extractor, LLM
from .llm.extractor import FieldExtractor
import os
from pathlib import Path
import litellm

# Get the root directory (where .env is located)
ROOT_DIR = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str
    OPENAI_API_VERSION: str
    OPENAI_DEPLOYMENT_NAME: str
    OPENAI_API_TYPE: str
    
    # Service URLs
    NER_SERVICE_URL: str
    FRONTEND_URL: str
    BACKEND_URL: str
    PYTHON_SERVICE_URL: str
    DOCKER_FRONTEND_URL: str
    DOCKER_BACKEND_URL: str
    DOCKER_PYTHON_SERVICE_URL: str
    DOCKER_NER_SERVICE_URL: str
    
    # Environment
    IS_DOCKER: bool = False
    NODE_ENV: str = "development"
    
    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = 'utf-8'
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name == "IS_DOCKER":
                return os.getenv("DOTNET_RUNNING_IN_CONTAINER") == "true"
            return raw_val

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