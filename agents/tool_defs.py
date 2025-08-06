import os
import requests
import json
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool

@tool
def brave_web_search(query: str) -> str:
    """
    Performs a web search using the Brave Search API to find up-to-date information.
    Use this for questions about current events, stock prices, or anything you don't have in your local knowledge base.
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return "Error: BRAVE_API_KEY not found."

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    params = {"q": query}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()
        
        if not results.get("web", {}).get("results"):
            return "No results found."

        # Format the results into a single string for the agent
        output = ""
        for i, item in enumerate(results["web"]["results"][:4]): # Get top 4 results
            output += f"Result {i+1}: {item.get('title')}\n"
            output += f"Snippet: {item.get('description')}\n"
            output += f"URL: {item.get('url')}\n---\n"
        return output
        
    except Exception as e:
        return f"An error occurred during web search: {e}"

def get_tools(vectorstore):
    """
    Creates and returns a list of tools for the agent to use.
    """
    # The RAG tool for searching local documents remains the same
    retriever = vectorstore.as_retriever()
    document_retriever_tool = create_retriever_tool(
        retriever,
        "local_document_search",
        "Searches and returns relevant information from your local knowledge base of documents. Use this for questions about salary guides, company policies, or resumes."
    )
    
    # We now return our custom brave_web_search tool
    return [brave_web_search, document_retriever_tool]