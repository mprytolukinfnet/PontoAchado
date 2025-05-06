from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance

from call_llm import generate_embeddings

import os
from dotenv import load_dotenv
from pathlib import Path

import pandas as pd

# Load environment variables
load_dotenv(dotenv_path=Path("../../.env"))

QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_HOST = os.environ.get("QDRANT_HOST")

model = None

# Qdrant cloud client initialization
qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)


# Function to create a Qdrant collection
def create_collection(collection_name, size=768):
    collection_exists = qdrant_client.collection_exists(collection_name)
    if collection_exists:
        print(f"Skipping collection `{collection_name}` creation as it already exists.")
    else:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )


# Function to upload vectors to Qdrant
def upload_vectors(atividade_id, vectors, collection_name):
    # Prepare and insert points into the collection
    points = [
        PointStruct(
            id=int(atividade_id),
            vector=vector.tolist(),
            # payload=metadata[idx]
        )
        for idx, vector in enumerate(vectors)
    ]

    qdrant_client.upsert(collection_name=collection_name, points=points)


def process_atividade(id, content, collection_name, model_origin):
    if model_origin == "huggingface":
        doc_vectors = model.encode([content], convert_to_tensor=True)
    elif model_origin == "gemini":
        doc_vectors = generate_embeddings(content)
    else:
        raise ValueError("Unknown `model_origin` parameter")

    # Upload vectors
    upload_vectors(id, doc_vectors, collection_name)


def create_vectors(df, collection_name, model_origin="gemini"):
    """
    model_origin = huggingface (for self host) or gemini
    """
    if model_origin == "huggingface":
        from sentence_transformers import SentenceTransformer
        global model
        model = SentenceTransformer(
            "stjiris/bert-large-portuguese-cased-legal-mlm-nli-sts-v1"
        )
    for idx, atividade_row in df.iterrows():
        atividade_id = atividade_row.codigo_atividade
        content = atividade_row.atividade
        process_atividade(atividade_id, content, collection_name, model_origin)


if __name__ == "__main__":
    create_collection("atividades", 768)
    atividades = pd.read_csv('../data/atividades.csv')
    create_vectors(atividades, collection_name="atividades", model_origin="gemini")
