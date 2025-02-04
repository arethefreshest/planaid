from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"  # Default to smaller model
    TEMPERATURE: float = 0.3  # Lower temperature for more consistent results
    MAX_TOKENS: int = 2000  # Token limit
    
    class Config:
        env_file = ".env"

settings = Settings() 