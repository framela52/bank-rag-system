from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any
from pathlib import Path
import json

class EmbeddingBuilder:
    def __init__(self, embedding_model: str = "intfloat/multilingual-e5-large"):
        """HuggingFace embeddings"""
        self.model_name = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"Загружена модель: {embedding_model}")
    
    def create_vector_store(self, chunks: List[Dict], metadata_key: str = "metadata") -> FAISS:
        texts = [chunk["text"] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = chunk.get(metadata_key, {}).copy()
            metadata['id'] = chunk['id']
            metadatas.append(metadata)
    
        print(f"Создание FAISS из {len(texts)} чанков...")
        vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        print(f"FAISS создан: {vector_store.index.ntotal} векторов")
        return vector_store
    
    def save_vector_store(self, vector_store: FAISS, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(path))
        print(f"Сохранено: {path}")
    
    def load_vector_store(self, path: Path) -> FAISS:
        return FAISS.load_local(str(path), self.embeddings, allow_dangerous_deserialization=True)
