from fastapi import FastAPI
from Routes import base,chat
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()  

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.include_router(base.base_router)
app.include_router(chat.chat_router)