from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    APP_NAME:str
    APP_VERSION:str
    
    
    GOOGLE_API_KEY : str
    FASTAPI_URL : str
    
    class Config:
        env_file = ".env"
        
def get_settings():
    return Settings()