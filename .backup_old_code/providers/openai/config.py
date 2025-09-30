from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class OpenAIConfig:
    """Configuração específica para OpenAI."""
    model: str = "gpt-4"
    reasoning_effort: str = "minimal"
    verbosity: str = "low"
    api_key: Optional[str] = None
    client: Optional[Any] = None