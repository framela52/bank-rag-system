from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Пути
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_stores"
    
    # Модели
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    LLM_MODEL: str = "mistral"  # для Ollama
    USE_OLLAMA: bool = True
    
    # Параметры чанкинга
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Ollama
    OLLAMA_BASE_URL: str = "http://ollama:11434" 
      
    # Параметры генерации
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 1000
    
    # GigaChat
    GIGACHAT_CLIENT_ID: str = ""
    GIGACHAT_CLIENT_SECRET: str = ""
    GIGACHAT_MODEL: str = "GigaChat:latest"
    
    # RAG настройки
    RETRIEVAL_K: int = 5
    USE_COMPRESSION: bool = True
    
    USE_SELF_QUERY: bool = True
    USE_MULTI_QUERY: bool = True
    USE_RERANK: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()