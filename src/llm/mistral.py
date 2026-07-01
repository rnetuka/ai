import os

from langchain_core.language_models import BaseChatModel
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr


def Mistral() -> BaseChatModel:
    if 'MISTRAL_API_KEY' not in os.environ:
        raise Exception('Mistral API key missing')
    api_key = SecretStr(os.environ['MISTRAL_API_KEY'])
    return ChatMistralAI(model_name='mistral-small-latest', api_key=api_key)
