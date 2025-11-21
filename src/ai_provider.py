"""
Multi-provider AI client supporting OpenAI, Anthropic, and Grok (xAI).
Provides a unified interface for making LLM API calls with parallel execution support.
"""

import os
import json
import asyncio
import time
import concurrent.futures
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env')
load_dotenv(env_path)


class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROK = "grok"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: AIProvider
    model_id: str
    max_tokens: int = 4000
    temperature: float = 0.2


@dataclass
class UsageStats:
    """Track token usage and timing for API calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    elapsed_time: float = 0.0
    model: str = ""
    provider: str = ""

    def __add__(self, other: 'UsageStats') -> 'UsageStats':
        return UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            elapsed_time=self.elapsed_time + other.elapsed_time,
            model=self.model or other.model,
            provider=self.provider or other.provider
        )


# Model mappings for each provider
PROVIDER_MODELS = {
    AIProvider.OPENAI: {
        "fast": "gpt-4o-mini",
        "standard": "gpt-4o",
        "powerful": "gpt-4o",
    },
    AIProvider.ANTHROPIC: {
        "fast": "claude-sonnet-4-5-20250929",
        "standard": "claude-sonnet-4-5-20250929",
        "powerful": "claude-sonnet-4-5-20250929",
    },
    AIProvider.GROK: {
        "fast": "grok-4-fast-reasoning",
        "standard": "grok-4-fast-reasoning",
        "powerful": "grok-4-fast-reasoning",
    },
}


class AIClient:
    """Unified AI client supporting multiple providers."""

    def __init__(self, provider: AIProvider = AIProvider.OPENAI):
        self.provider = provider
        self._client = None
        self._initialize_client()
        # Track cumulative usage stats
        self.total_usage = UsageStats()
        self.last_usage = UsageStats()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == AIProvider.OPENAI:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self._client = openai.OpenAI(api_key=api_key)

        elif self.provider == AIProvider.ANTHROPIC:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self._client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == AIProvider.GROK:
            import openai
            api_key = os.getenv('GROK_API_KEY') or os.getenv('XAI_API_KEY')
            if not api_key:
                raise ValueError("GROK_API_KEY or XAI_API_KEY not found in environment")
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )

    def get_model(self, tier: str = "standard") -> str:
        """Get the appropriate model for the current provider."""
        return PROVIDER_MODELS[self.provider].get(tier, PROVIDER_MODELS[self.provider]["standard"])

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model_tier: str = "standard",
        temperature: float = 0.2,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Make a chat completion request to the configured provider.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_tier: 'fast', 'standard', or 'powerful'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt (used differently per provider)

        Returns:
            The assistant's response text
        """
        model = self.get_model(model_tier)
        start_time = time.time()

        if self.provider == AIProvider.ANTHROPIC:
            result, usage = self._anthropic_completion(messages, model, temperature, max_tokens, system_prompt)
        else:
            # OpenAI and Grok use the same API format
            result, usage = self._openai_completion(messages, model, temperature, max_tokens, system_prompt)

        # Update usage stats
        usage.elapsed_time = time.time() - start_time
        usage.model = model
        usage.provider = self.provider.value
        self.last_usage = usage
        self.total_usage = self.total_usage + usage

        return result

    def reset_usage(self):
        """Reset cumulative usage stats."""
        self.total_usage = UsageStats()
        self.last_usage = UsageStats()

    def _openai_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str]
    ) -> Tuple[str, UsageStats]:
        """Make completion request using OpenAI-compatible API."""
        formatted_messages = []

        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        response = self._client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Extract usage stats
        usage = UsageStats()
        if hasattr(response, 'usage') and response.usage:
            usage.input_tokens = response.usage.prompt_tokens or 0
            usage.output_tokens = response.usage.completion_tokens or 0
            usage.total_tokens = response.usage.total_tokens or 0

        return response.choices[0].message.content, usage

    def _anthropic_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str]
    ) -> Tuple[str, UsageStats]:
        """Make completion request using Anthropic API."""
        formatted_messages = []

        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        # Extract usage stats
        usage = UsageStats()
        if hasattr(response, 'usage') and response.usage:
            usage.input_tokens = response.usage.input_tokens or 0
            usage.output_tokens = response.usage.output_tokens or 0
            usage.total_tokens = usage.input_tokens + usage.output_tokens

        return response.content[0].text, usage

    @staticmethod
    def run_parallel(tasks: List[Callable[[], Any]], max_workers: int = 3) -> List[Any]:
        """
        Run multiple tasks in parallel using ThreadPoolExecutor.

        Args:
            tasks: List of callable functions
            max_workers: Maximum concurrent workers

        Returns:
            List of results in same order as tasks
        """
        results = [None] * len(tasks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(task): idx for idx, task in enumerate(tasks)}

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Error: {str(e)}"

        return results


def get_available_providers() -> List[str]:
    """Return list of providers that have API keys configured."""
    available = []

    if os.getenv('OPENAI_API_KEY'):
        available.append("OpenAI (GPT-4o)")

    if os.getenv('ANTHROPIC_API_KEY'):
        available.append("Anthropic (Claude)")

    if os.getenv('GROK_API_KEY') or os.getenv('XAI_API_KEY'):
        available.append("Grok (xAI)")

    return available


def provider_from_name(name: str) -> AIProvider:
    """Convert display name to AIProvider enum."""
    name_lower = name.lower()
    if "openai" in name_lower or "gpt" in name_lower:
        return AIProvider.OPENAI
    elif "anthropic" in name_lower or "claude" in name_lower:
        return AIProvider.ANTHROPIC
    elif "grok" in name_lower or "xai" in name_lower:
        return AIProvider.GROK
    return AIProvider.OPENAI  # Default
