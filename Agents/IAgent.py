from Helpers.configs import get_settings
from abc import ABC
from langchain_google_genai import ChatGoogleGenerativeAI

class IAgent(ABC):
    def __init__(self):
        self.app_settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            max_tokens=500,
            api_key=self.app_settings.GOOGLE_API_KEY,
            max_retries=2,
            )
       
    