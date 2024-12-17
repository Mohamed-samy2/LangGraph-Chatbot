from .IAgent import IAgent
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
import operator

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    categories: Annotated[dict[str,str],operator.__or__] = {}
    
class categories(BaseModel):
    """Identify the user categories"""
    
    size:str=Field(
        description="if the user answered with his size, if not make it -1"
    )
    color:str=Field(
        description="if the user answer with his color, if not make it -1"
    )
    
    price:str=Field(
        description=" if the user answer about specific price, if not make it -1 "
    )
    

class ClothesAgent(IAgent):
    def __init__(self,memory):
        super(ClothesAgent).__init__()
        workflow = StateGraph(AgentState)
        
        workflow.add_node("ask_size",self.ask_size)
        workflow.add_node("identify_size",self.identify_size)
        workflow.add_node("ask_color",self.ask_color)
        workflow.add_node("identify_color",self.identify_color)
        
        workflow.add_edge(START,"ask_size")
        workflow.add_edge("ask_size",'identify_size')
        workflow.add_edge('identify_size','ask_color')
        workflow.add_edge('ask_color','identify_color')
        workflow.add_edge('identify_color',END)
        
        
        self.graph = workflow.compile(
            interrupt_after=['ask_size','ask_color'],
            checkpointer=memory
            )
    
    
    def ask_color(self,state:AgentState):
        message = state['messages'][-2]
        
        prompt = """You are a shop customer service you want to know the user color \n
                    about the specific category he want \n
                    ask him politely about his color\n
                    """
        
        grade_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("human", "user message:\n\n {message}")]
        )
        
        evaluator = grade_prompt | self.llm
        result = evaluator.invoke(
            message
        )
        return {"messages" : result}
    
    def identify_color(self,state:AgentState):
        structredllm = self.llm.with_structured_output(categories)
        
        message = state['messages'][-1]
        prompt = """You are a shop customer service you want to know the user color \n
                    based on the user answer \n
                    identify the user color if yes tell me the specific color if not return -1 \n
                    """
        
        grade_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("human", "user message:\n\n {message}")]
        )
        evaluator = grade_prompt | structredllm
        result = evaluator.invoke(
            message
        )
        
        return {'categories':{"color" : result.color}}
    
    def ask_size(self,state:AgentState):
        message = state['messages'][-1]
        
        prompt = """You are a shop customer service you want to know the user size \n
                    about the specific category he want \n
                    ask him politely about his size\n
                    """
        
        grade_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("human", "user message:\n\n {message}")]
        )
        
        evaluator = grade_prompt | self.llm
        result = evaluator.invoke(
            message
        )
        return {"messages" : result}
    
    def identify_size(self,state:AgentState):
        structredllm = self.llm.with_structured_output(categories)
        
        message = state['messages'][-1]
        prompt = """You are a shop customer service you want to know the user size \n
                    based on the user answer \n
                    identify the user size if yes tell me the specific size if not return -1 \n
                    """
        
        grade_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("human", "user message:\n\n {message}")]
        )
        evaluator = grade_prompt | structredllm
        result = evaluator.invoke(
            message
        )
        
        return {'categories':{"size" : result.size}}
    
    
    
        
    
    
        