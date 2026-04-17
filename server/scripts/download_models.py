"""
Usage:
    python scripts/download_models.py
"""
from __future__ import annotations

import importlib
import logging
import os

from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import constants

LOGGER = logging.getLogger("download_models")
SERVER_DIR = Path(__file__).resolve().parents[1]

def _get_model_name(env_name: str) -> str:
    value = os.environ.get(env_name)
    model_name = str(value or "").strip()
    if not model_name:
        raise ValueError(f"{env_name} is empty")
    return model_name


def download_core_models() -> None:
    embedding_model = _get_model_name("NEXUSRAG_EMBEDDING_MODEL")
    reranker_model = _get_model_name("NEXUSRAG_RERANKER_MODEL")
    gliner_model = _get_model_name("NEXUSRAG_KG_GLINER_MODEL")
    relation_model = _get_model_name("NEXUSRAG_KG_RELATION_MODEL")

    from sentence_transformers import SentenceTransformer, CrossEncoder
    gliner_module = importlib.import_module("gliner")
    transformers_module = importlib.import_module("transformers")
    GLiNER = getattr(gliner_module, "GLiNER")
    AutoTokenizer = getattr(transformers_module, "AutoTokenizer")
    AutoModelForSeq2SeqLM = getattr(transformers_module, "AutoModelForSeq2SeqLM")
    
    print(f"[1/4] Downloading embedding model: {embedding_model}")
    SentenceTransformer(embedding_model)
    print("      Done.")

    print(f"[2/4] Downloading reranker model: {reranker_model}")
    CrossEncoder(reranker_model)
    print("      Done.")
    
    print(f"[3/4] Downloading GLiNER model: {gliner_model}")
    GLiNER.from_pretrained(gliner_model)
    print("      Done.")
    
    
    print(f"[4/4] Downloading mREBEL model: {relation_model}")
    AutoTokenizer.from_pretrained(relation_model, use_fast=True)
    AutoModelForSeq2SeqLM.from_pretrained(relation_model)
    print("      Done.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    load_dotenv(SERVER_DIR / ".env", override=False)
    download_core_models()
    print("=" * 50)
    print(f"Các model được tải về tại : {constants.HF_HOME}")
    
    
