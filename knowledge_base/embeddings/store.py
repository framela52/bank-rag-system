from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import List, Dict, Optional
import json
from .builder import EmbeddingBuilder

class VectorStoreManager:
    """Менеджер для работы с векторными хранилищами"""
    
    def __init__(self, base_path: Path = Path("data/vector_stores")):
        self.base_path = base_path
        self.embedding_builder = EmbeddingBuilder()
        self.vector_stores = {}
    
    def build_from_chunks(self, chunks_file: Path, store_name: str) -> bool:
        """
        Построение векторного хранилища из файла с чанками
        
        Args:
            chunks_file: Путь к JSON файлу с чанками
            store_name: Имя векторного хранилища
            
        Returns:
            bool: Успешность операции
        """
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Создаем векторное хранилище
            vector_store = self.embedding_builder.create_vector_store(chunks)
            
            # Сохраняем
            store_path = self.base_path / store_name
            self.embedding_builder.save_vector_store(vector_store, store_path)
            
            self.vector_stores[store_name] = vector_store
            return True
            
        except Exception as e:
            print(f"Ошибка при создании векторного хранилища {store_name}: {e}")
            return False
    
    def load_store(self, store_name: str) -> Optional[FAISS]:
        """Загрузка существующего векторного хранилища"""
        store_path = self.base_path / store_name
        
        if store_path.exists():
            vector_store = self.embedding_builder.load_vector_store(store_path)
            self.vector_stores[store_name] = vector_store
            return vector_store
        
        return None
    
    def get_store(self, store_name: str) -> Optional[FAISS]:
        """Получение векторного хранилища по имени"""
        if store_name in self.vector_stores:
            return self.vector_stores[store_name]
        
        return self.load_store(store_name)
    
    def compare_stores(self, store_names: List[str], query: str, k: int = 5) -> Dict:
        """
        Сравнение разных векторных хранилищ по одному запросу
        
        Args:
            store_names: Список имен хранилищ
            query: Тестовый запрос
            k: Количество результатов
            
        Returns:
            Dict: Результаты поиска для каждого хранилища
        """
        results = {}
        
        for store_name in store_names:
            store = self.get_store(store_name)
            if store:
                docs = store.similarity_search_with_score(query, k=k)
                results[store_name] = [
                    {
                        "text": doc[0].page_content[:200] + "...",
                        "score": doc[1],
                        "metadata": doc[0].metadata
                    }
                    for doc in docs
                ]
        
        return results