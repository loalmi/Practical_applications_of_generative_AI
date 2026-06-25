"""
llm_client.py — упрощённый клиент для LLM
"""

import os
import json
import re
from typing import Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T", bound=BaseModel)


def get_model() -> str:
    """Возвращает имя модели из .env"""
    return os.getenv("LLM_MODEL", "deepseek-v4-flash")


def get_client() -> OpenAI:
    """Создаёт клиент OpenAI"""
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_AUTH_TOKEN") or os.getenv("OPENAI_API_KEY")
    
    if not base_url:
        raise ValueError("LLM_BASE_URL не задан в .env")
    if not api_key:
        raise ValueError("LLM_AUTH_TOKEN или OPENAI_API_KEY не задан в .env")
    
    return OpenAI(base_url=base_url, api_key=api_key)


def ask_llm(
    prompt: str,
    response_model: Type[T],
    temperature: float = 0.3,
    max_retries: int = 2
) -> T:
    """
    Универсальная функция для запроса к LLM с парсингом в Pydantic
    """
    client = get_client()
    model = get_model()
    
    # Добавляем инструкцию про JSON
    system_msg = "Ты — помощник. Отвечай строго в формате JSON. Без пояснений."
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            
            content = resp.choices[0].message.content
            
            # Пробуем найти JSON в ответе
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return response_model(**data)
            else:
                raise ValueError(f"JSON не найден в ответе: {content[:200]}")
                
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  ⚠️ Попытка {attempt+1} не удалась: {e}")
    
    raise RuntimeError("Не удалось получить ответ от LLM")


# Простая обёртка для обратной совместимости
class ChatCompletionsProxy:
    def __init__(self, client: OpenAI):
        self._client = client
    
    def create(self, model: str, messages: list[dict], response_model: Type[T], **kwargs):
        """Имитирует интерфейс из семинаров"""
        # Извлекаем пользовательский запрос
        user_content = None
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break
        
        if not user_content:
            raise ValueError("Не найден пользовательский запрос")
        
        return ask_llm(user_content, response_model, temperature=kwargs.get("temperature", 0.3))


def make_client():
    """Возвращает объект с интерфейсом как в семинарах"""
    return type('Client', (), {
        'chat': type('Chat', (), {
            'completions': ChatCompletionsProxy(get_client())
        })()
    })()