import os

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr


def Gemini() -> BaseChatModel:
    if 'GOOGLE_API_KEY' not in os.environ:
        raise Exception('Google API key missing')
    api_key = SecretStr(os.environ['GOOGLE_API_KEY'])
    return ChatGoogleGenerativeAI(model='gemini-3.5-flash', api_key=api_key)
