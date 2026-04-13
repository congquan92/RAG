import os
import sys
from dotenv import load_dotenv
load_dotenv()

def download_models():
    embedding_model = os.environ.get("NEXUSRAG_EMBEDDING_MODEL")
    reranker_model = os.environ.get("NEXUSRAG_RERANKER_MODEL")

    from sentence_transformers import SentenceTransformer, CrossEncoder

    print(f"[1/2] Downloading embedding model: {embedding_model}")
    SentenceTransformer(embedding_model)
    print(f"      Done.")

    print(f"[2/2] Downloading reranker model: {reranker_model}")
    CrossEncoder(reranker_model)
    print(f"      Done.")

    print("\nAll models downloaded successfully.")

if __name__ == "__main__":
    download_models()
