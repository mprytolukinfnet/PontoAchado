from qdrant_client import QdrantClient
from ai.call_llm import generate_embeddings
import sys

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path('../../.env'))
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
QDRANT_HOST = os.environ.get('QDRANT_HOST')

# Initialize Qdrant client and SentenceTransformer model
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)
model = None

def setup_local_model():
    '''
    If using local model to search for similar vectors, it is necessary to instatiate it before using.
    '''
    from sentence_transformers import SentenceTransformer
    global model
    model = SentenceTransformer('stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1')

def search_similar_atividades(query, n=5, collection_name="atividades", model_origin="gemini"):
    """
    Search for the n most similar listings to the query string.
    
    Args:
    - query (str): The user's search query.
    - n (int): Number of similar listings to return.
    
    Returns:
    - List[int]: IDs of the most similar listings.
    """
    # Transform query into vector
    if model_origin == "huggingface":
        if model == None:
            setup_local_model()
        query_vector = model.encode([query], convert_to_tensor=True).tolist()[0]
    elif model_origin == "gemini":
        try:
            query_vector = generate_embeddings(query).tolist()[0]
        except:
            if 'streamlit' in sys.modules:
                st.warning("Erro na chamada à API do Modelo de Embeddings")
    else:
        raise ValueError("Unknown `model_origin` parameter value")
    
    # Perform search in Qdrant
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=n
    )
    
    # Extract and return listing IDs from search results
    return [hit.id for hit in search_result]

if __name__ == "__main__":
    query = "Quero uma casa com piscina e um lindo jardim"
    similar_listings = search_similar_listings(query, n=5)
    print("Most similar listings:", similar_listings)
