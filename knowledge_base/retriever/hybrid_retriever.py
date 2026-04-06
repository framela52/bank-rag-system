from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pathlib import Path
import json
from typing import List, Optional


class HybridRetriever:
    def __init__(self, product_type_filter: str = None):
        print("ГИБРИДНЫЙ РЕТРИВЕР (FAISS + BM25)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        self.faiss = self._load_faiss()
        self.bm25 = self._load_bm25()
        self.product_type_filter = product_type_filter

        if self.faiss and self.bm25:
            print("Гибрид готов!")

    def _load_faiss(self):
        
        store_path = "data/vector_stores/faiss_recursive"
           
        try:
            store = FAISS.load_local(
                str(store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"FAISS загружен: {store.index.ntotal} векторов")            
            return store
        except Exception as e:
            print(f"Ошибка загрузки FAISS: {e}")
            return None
    
    def _load_bm25(self):
        
        chunks_file = "data/processed/chunks_recursive.json"
         
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Преобразуем в Document объекты
            documents = []
            for chunk in chunks:
                metadata = chunk.get('metadata', {})
                # ДОБАВЛЯЕМ id из верхнего уровня в метаданные
                metadata['id'] = chunk['id']  
                doc = Document(
                    page_content=chunk['text'],
                    metadata=metadata
                )
                documents.append(doc)
            
            retriever = BM25Retriever.from_documents(documents)
            retriever.k = 10
            print(f"BM25 загружен: {len(documents)} текстов")
            return retriever
        except Exception as e:
            print(f"Ошибка загрузки BM25: {e}")
            return None
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Только FAISS поиск."""
        if self.faiss:
            try:
                docs = self.faiss.similarity_search(query, k=k)
                return docs
            except Exception as e:
                print(f"Ошибка FAISS поиска: {e}")
                return []
        return []
    
    def bm25_search(self, query: str, k: int = 5) -> List[Document]:
        """Только BM25 поиск."""
        if self.bm25:
            try:
                docs = self.bm25.invoke(query)
                return docs[:k] if len(docs) > k else docs
            except Exception as e:
                print(f"Ошибка BM25 поиска: {e}")
                return []
        return []
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Гибридный поиск: FAISS + BM25 с весами."""
        results = {}
        
        # FAISS 
        if self.faiss:
            try:
                faiss_docs = self.faiss.similarity_search_with_score(query, k=10)
                for i, (doc, score) in enumerate(faiss_docs):
                    # Чем меньше score, тем лучше (расстояние), поэтому преобразуем
                    similarity = 1.0 / (1.0 + score)
                    doc_id = doc.metadata.get('id', f'faiss_{i}')
                    results[doc_id] = {
                        'doc': doc, 
                        'score': 0.6 * similarity, 
                        'source': 'faiss'
                    }
            except Exception as e:
                print(f"Ошибка FAISS поиска: {e}")
        
        # BM25 
        if self.bm25:
            try:
                bm25_docs = self.bm25.invoke(query)
                for i, doc in enumerate(bm25_docs[:10]):
                    # BM25 не даёт нормализованную оценку, используем позицию
                    bm25_score = 1.0 / (i + 1)
                    doc_id = doc.metadata.get('id', f'bm25_{i}')
                    old_score = results.get(doc_id, {}).get('score', 0)
                    results[doc_id] = {
                        'doc': doc,
                        'score': old_score + 0.4 * bm25_score,
                        'source': results.get(doc_id, {}).get('source', 'bm25')
                    }
            except Exception as e:
                print(f"Ошибка BM25 поиска: {e}")
        
        # Сортировка по score и фильтрация
        sorted_results = sorted(results.values(), key=lambda x: x['score'], reverse=True)
        
        # Фильтрация по типу продукта
        if self.product_type_filter:
            filtered = []
            for r in sorted_results:
                if r['doc'].metadata.get('product_type') == self.product_type_filter:
                    filtered.append(r['doc'])
            return filtered[:k]
        
        return [r['doc'] for r in sorted_results[:k]]
    
    def test_hybrid(self, queries: List[str]):
        """Тестирование гибридного поиска."""
        print("\n" + "="*80)
        print("ТЕСТ ГИБРИДНОГО РЕТРИВЕРА")
        print("="*80)
        
        for query in queries:
            print(f"\nЗапрос: {query}")
            
            # FAISS only
            faiss_docs = self.similarity_search(query, k=3)
            print(f"  FAISS: {[doc.metadata.get('product_type', 'unknown') for doc in faiss_docs]}")
            
            # BM25 only
            bm25_docs = self.bm25_search(query, k=3)
            print(f"  BM25:  {[doc.metadata.get('product_type', 'unknown') for doc in bm25_docs]}")
            
            # Hybrid
            hybrid_docs = self.hybrid_search(query, k=3)
            print(f"  HYBRID:{[doc.metadata.get('product_type', 'unknown') for doc in hybrid_docs]}")
            print(f"  Тексты первых результатов:")
            for doc in hybrid_docs[:2]:
                text_preview = doc.page_content[:100].replace('\n', ' ')
                print(f"    - {text_preview}...")
        
        print("="*80)


if __name__ == "__main__":
    # Тестирование
    retriever = HybridRetriever()
    
    test_queries = [
        "депозит минимальная сумма",
        "процентная ставка по депозиту",
        "ипотека первоначальный взнос",
        "потребительский кредит максимальная сумма"
    ]
    
    retriever.test_hybrid(test_queries)