from fastapi import APIRouter , status, Request,Depends
from fastapi.responses import JSONResponse
from Helpers.Enums.ResponseEnum import Response
from Agents import MainAgent
from langgraph.checkpoint.memory import MemorySaver

chat_router = APIRouter()
llm = MainAgent(memory=MemorySaver())

    

@chat_router.post("/chat")
async def process_endpoint(request:Request,query:str):
    
    config = {'configurable': {'thread_id' : 1}}
    
    response = llm.run(config=config,message=query)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'response':Response.CHAT_SUCCESS.value,
            'query':response,
        }
    )

