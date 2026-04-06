from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from typing import List
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from knowledge_base.retriever.hybrid_retriever import HybridRetriever


class HybridRetrieverWithCompression(HybridRetriever):
    def __init__(self, product_type_filter: str = None,
                 use_compression: bool = True,
                 compression_k: int = 3,
                 model_name: str = None):
        super().__init__(product_type_filter)

        self.use_compression = use_compression
        self.compression_k = compression_k
        
        if self.use_compression:
            self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
            print("Компрессия через эмбеддинги (без LLM)")

    def _get_query_embedding(self, query: str):
        """Получает эмбеддинг запроса."""
        return self.embeddings.embed_query(query)
    
    def _get_doc_embedding(self, doc: Document):
        """Получает эмбеддинг документа."""
        return self.embeddings.embed_query(doc.page_content)
    
    def _calculate_similarity(self, query_emb, doc_emb):
        """Вычисляет косинусное сходство."""
        query_emb = np.array(query_emb)
        doc_emb = np.array(doc_emb)
        return np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Гибридный поиск с фильтрацией по релевантности."""
        # Получаем больше документов (в 2-3 раза больше)
        docs = super().hybrid_search(query, k=k * 3)
        
        if not docs:
            return []
        
        # Применяем фильтрацию по релевантности (компрессия без LLM)
        if self.use_compression:
            try:
                query_emb = self._get_query_embedding(query)
                
                # Вычисляем релевантность каждого документа
                scored_docs = []
                for doc in docs:
                    doc_emb = self._get_doc_embedding(doc)
                    similarity = self._calculate_similarity(query_emb, doc_emb)
                    scored_docs.append((doc, similarity))
                
                # Сортируем по релевантности
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                # Возвращаем топ-k документов
                return [doc for doc, score in scored_docs[:k]]
            except Exception as e:
                print(f"Ошибка фильтрации: {e}")
                return docs[:k]
        
        return docs[:k]