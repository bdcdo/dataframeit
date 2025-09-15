from dataclasses import dataclass
from typing import Optional


@dataclass
class LangChainConfig:
    """Configuração específica para LangChain."""
    model: str = "gemini-2.5-flash"
    provider: str = "google_genai"
    temperature: float = 0
    api_key: Optional[str] = None