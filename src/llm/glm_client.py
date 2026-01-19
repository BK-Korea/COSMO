"""GLM-4.7 client integration (following NOVA pattern)"""
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from config.config import settings


class GLMClient:
    """Client for interacting with Zhipu AI's GLM models"""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize GLM client

        Args:
            model: Model name (default from settings)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model or settings.chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LangChain ChatOpenAI with GLM endpoint
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=settings.glm_api_key,
            openai_api_base=settings.glm_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_llm(self) -> BaseChatModel:
        """Get the underlying LangChain LLM instance"""
        return self.llm

    def chat(
        self,
        messages: List[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Simple chat completion

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt

        Returns:
            Response text
        """
        langchain_messages = []

        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))

        response = self.llm.invoke(langchain_messages)
        return response.content

    def simple_query(self, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple single-turn query

        Args:
            query: User query
            system_prompt: Optional system prompt

        Returns:
            Response text
        """
        messages = [{"role": "user", "content": query}]
        return self.chat(messages, system_prompt=system_prompt)


def create_glm_client(
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> GLMClient:
    """Factory function to create GLM client"""
    return GLMClient(model=model, temperature=temperature, max_tokens=max_tokens)
