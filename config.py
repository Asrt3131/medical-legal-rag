import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

# تنظیمات Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3