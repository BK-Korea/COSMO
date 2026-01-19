"""Configuration management for COSMO (following NOVA pattern)"""
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings using Pydantic (NOVA-style)"""

    # GLM Configuration
    glm_api_key: str = Field(..., alias="GLM_API_KEY")
    glm_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/",
        alias="GLM_BASE_URL"
    )

    # OpenAI Configuration (for embeddings)
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # Model Configuration
    chat_model: str = Field(default="glm-4-flash", alias="CHAT_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="EMBEDDING_MODEL"
    )
    embedding_provider: Literal["openai", "glm"] = Field(
        default="openai",
        alias="EMBEDDING_PROVIDER"
    )

    # Vector Store Configuration
    vector_store_type: Literal["chroma"] = "chroma"
    chroma_persist_dir: Path = Field(default=Path("data/vectordb"))

    # Yahoo Finance Configuration
    yfinance_cache_dir: Path = Field(default=Path("data/yfinance_cache"))

    # Chunking Configuration (following NOVA)
    chunk_size: int = Field(default=1000, description="Token count per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")

    # Retrieval Configuration
    retrieval_top_k: int = Field(default=10, description="Number of chunks to retrieve")

    # Quality Configuration (like NOVA's quality evaluator)
    quality_threshold: float = Field(
        default=8.0,
        ge=0.0,
        le=10.0,
        description="Minimum quality score (0-10)"
    )

    # Data Paths
    data_dir: Path = Field(default=Path("data"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def model_post_init(self, __context):
        """Create directories after initialization"""
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.yfinance_cache_dir.mkdir(parents=True, exist_ok=True)


# Create singleton instance
settings = Settings()
