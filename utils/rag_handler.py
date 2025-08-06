import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

def setup_vector_store(embedding_model, docs_path="documents"):
    """
    Loads docs, splits them, and builds a vector store using a provided embedding model.
    """
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return None
    
    loader = PyPDFDirectoryLoader(docs_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
    return vectorstore

def retrieve_context(query, vectorstore):
    """
    Retrieves relevant context from the vector store based on a query.
    """
    if not vectorstore:
        return None
        
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)
    
    if not retrieved_docs:
        return None
        
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

def get_targeted_salary_estimation(resume_data, target_role, target_location, vectorstore, llm, response_mode="Detailed"):
    """
    Compares a resume profile against a target role and uses RAG to estimate salary.
    """
    rag_query = f"What is the salary range for a '{target_role}' in {target_location}?"
    context = retrieve_context(rag_query, vectorstore)
    
    if not context:
        return "Could not find any relevant salary data in the knowledge base for this target role."

    experience = resume_data.get("total_experience_years", "Not specified")
    skills = resume_data.get("technical_skills", {}).get("libraries_and_technologies", [])
    
    # Define the instruction based on the selected mode
    if response_mode == "Concise":
        style_instruction = "Provide a concise, summary-level answer in 2-3 sentences."
    else:
        style_instruction = "Provide a detailed, in-depth analysis. Justify your reasoning with specific data points from the context."

    final_prompt = f"""
    You are an expert career coach and compensation analyst. Based on the following context from salary reports, provide a salary range for the target role and analyze the candidate's fit.
    {style_instruction}

    **Context from Salary Reports:**
    ---
    {context}
    ---

    **Target Role:**
    - Job Title: {target_role}
    - Location: {target_location}

    **Candidate's Profile (from their resume):**
    - Total Experience: {experience}
    - Key Skills: {', '.join(skills[:5])}
    """
    
    response = llm.invoke(final_prompt)
    return response.content