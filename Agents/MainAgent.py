from .IAgent import IAgent
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated,Literal
from pydantic import BaseModel
from langchain.tools import tool
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
import operator

class user_intent(BaseModel):
    """Identify the user intent"""
    
    faq:str=Field(
        description="if the user asked about the faq like the website policy or return days , 'yes' or 'no' "
    )
    category:str=Field(
        description="if the user asked about categories like clothes , 'yes' or 'no' "
    )
    unknown:str=Field(
        description=" if the user didn't ask about any thing or you don't know his intent or what he want, 'yes' or 'no' "
    )


class categories(BaseModel):
    """Identify the user categories"""
    
    size:str=Field(
        description="if the user answered with his size, if not make it -1"
    )
    color:str=Field(
        description="if the user answer with his color, if not make it -1"
    )

    
    

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    categories: Annotated[dict[str,str],operator.__or__] = {}


class MainAgent(IAgent):
    def __init__(self,memory):
        super().__init__()
        workflow = StateGraph(AgentState)
        workflow.add_node("greeting",self.greeting)
        workflow.add_node("dummy",self.dummy)
        workflow.add_node("unknown",self.unknown)
        workflow.add_node("ask_size",self.ask_size)
        workflow.add_node("identify_size",self.identify_size)
        workflow.add_node("faq",self.faq)
        workflow.add_node("ask_color",self.ask_color)
        workflow.add_node("identify_color",self.identify_color)
        workflow.add_node("finish",self.finish)
        
        
        workflow.add_edge(START,'greeting')
        workflow.add_edge('greeting','dummy')
        workflow.add_edge("unknown",'dummy')
        workflow.add_edge("ask_size",'identify_size')
        workflow.add_edge('identify_size','ask_color')
        workflow.add_edge("ask_color",'identify_color')
        workflow.add_edge("identify_color",'finish')
        workflow.add_edge("faq",'dummy')
        workflow.add_edge("finish",END)
        
        
        workflow.add_conditional_edges("dummy",
                                       self.user_intent,
                                       {
                                            "category": "ask_size",
                                            "faq": "faq",
                                            "unk": "unknown",
                                        })
        
        self.graph = workflow.compile(
            interrupt_after=['dummy','ask_size','ask_color'],
            checkpointer=memory
            )
    
    def run(self,config,message):
        
        current_values = self.graph.get_state(config)
        
        if 'messages' in current_values.values:
            _id = current_values.values['messages'][-1].id
            message = HumanMessage(content=message,id=_id)
            current_values.values['messages'][-1] = message
            self.graph.update_state(config , current_values.values)
            result = self.graph.invoke(None, config)
            return result['messages'][-1].content
        
        messages = [HumanMessage(content=message)]
        result = self.graph.invoke({'messages': messages}, config)
        return result['messages'][-1].content
    
    def greeting(self,state:AgentState):
        return {"messages":"Hello how can i help you today ?"}
    
    def user_intent(self,state:AgentState)-> Literal["category", "faq", "unk"]:
        
        structredllm = self.llm.with_structured_output(user_intent)
        
        message = state["messages"][-1]
        
        prompt = """You are a shop customer service that want to identify the user intent \n
                    check if the user asking about faq like website policy or return days \n
                    or if he ask about category like clothes  \n
                    or if he said an unknown something or general question\n
                    Provide a binary score 'yes' or 'no' to indicate the user intent"""
        
        grade_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("human", "question:\n\n {question}")]
        )
        
        evaluator = grade_prompt | structredllm
        result = evaluator.invoke(
            message
        )
        
        if result.category =='yes':
            return 'category'
        
        elif result.faq =='yes':
            return 'faq'
        else:
            return "unk"
    
    def unknown(self,state:AgentState):
        message = state["messages"][-1]
        
        prompt = """You are a shop customer service you want to know the user intent \n
                    but the user was talking about unknown something \n
                    ask him politely what he want again\n
                    """
        grade_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("human", "user message:\n\n {message}")]
        )
        evaluator = grade_prompt | self.llm
        result = evaluator.invoke(
            message
        )
        
        return {"messages":result}
        
    
    def dummy(self,state:AgentState):
        return {"messages":state['messages'][-1]}
        
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
        
        return {"categories":{'size':result.size}}
    
    def ask_color(self,state:AgentState):
        message = state['messages'][-2]
        
        prompt = """You are a shop customer service you want to know the user color \n
                    about the specific category he want \n
                    ask him politely about his color \n
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
        
        return {"categories":{"color" : result.color}}
    
    def faq(self,state:AgentState):
        message = state['messages'][-1]
        
        prompt = """You are a shop customer service you want to answer the FAQ questions \n
                    based on the user question \n
                    - if he ask about the return policy answer that is only 14 days \n
                    - if he asked about order days answer that from 2 to 4 days \n
                    and tell him can i help you in other thing in polite way \n
                    """
        grade_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("human", "user message:\n\n {message}")]
        )
        evaluator = grade_prompt | self.llm
        result = evaluator.invoke(
            message
        )
        return {'messages' : result}
        
    def finish(self,state:AgentState):
        size = state['categories']['size']
        color = state['categories']['color']
        
        cat = f""" user size : {size}\n
                user color : {color}\n
                """
        
        prompt = """You are a shop customer service \n
                    i want you to list the user preference in a good way \n                    
                """
        
        grade_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt), ("human", "user prefences:\n\n {cat}")]
        )
        evaluator = grade_prompt | self.llm
        result = evaluator.invoke(
            cat
        )
        
        return {'messages' : result}