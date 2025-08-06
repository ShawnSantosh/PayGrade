import os
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq

def get_agent_llm():
    """Returns the Groq LLM, which is suitable for agentic tasks."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
        
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0
    )

def create_agent_executor(llm, tools):
    """Creates the agent and the agent executor that runs it."""
    prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor