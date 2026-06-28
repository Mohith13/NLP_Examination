import os

# Define the absolute base directory of your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define where your databases will live inside the 'data' folder you just created
DATA_DIR = os.path.join(BASE_DIR, "data")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "bmw_metadata.db")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "bmw_vector_store")

# Define your AI and Embedding models
# Using 'llama3' to match exactly what you pulled earlier in the terminal
LLM_MODEL = "llama3" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ensure the data directory exists before any other script runs
os.makedirs(DATA_DIR, exist_ok=True)