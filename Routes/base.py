from fastapi import APIRouter , Depends,status
from fastapi.responses import JSONResponse
import os
from Helpers.configs import get_settings,Settings

base_router = APIRouter()

@base_router.get("/")
async def welcome(app_settings:Settings = Depends(get_settings)):
    app_settings = get_settings()
    
    app_name = app_settings.APP_NAME
    app_version= app_settings.APP_VERSION
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'query':f"Welcome to {app_name} How can i help you ?",
        }
    )