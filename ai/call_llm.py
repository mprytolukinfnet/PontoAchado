import google.generativeai as genai
import torch

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path("../.env"))
load_dotenv(dotenv_path=Path("../../.env"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Configuração da API Gemini
genai.configure(api_key=GEMINI_API_KEY)


# Function to call the Gemini API
def call_llm(prompt_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt_text)
    return response.text

# Function to call the Gemini API with streaming
async def call_llm_stream(prompt_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt_text, stream=True)
    
    # Iterate over the chunks from the response
    for chunk in response:
        yield chunk.text

# Function to call the Gemini API for embeddings
def generate_embeddings(text):
    result = genai.embed_content(model="models/text-embedding-004", content=[text])
    tensor = torch.tensor(result["embedding"])
    return tensor
