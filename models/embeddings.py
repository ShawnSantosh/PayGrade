# models/embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Initializes and returns the HuggingFace embedding model.
    """
    # This specifies the model to use. "all-MiniLM-L6-v2" is a great, lightweight default.
    model_name = "all-MiniLM-L6-v2"
    
    # The model will be downloaded automatically on the first run and cached for future use.
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    return embedding_model