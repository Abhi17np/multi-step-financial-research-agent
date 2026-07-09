from dotenv import load_dotenv
import os

load_dotenv()  # reads .env and loads its variables into the environment

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "research_agent_10k"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

MAX_VERIFIER_RETRIES = 2