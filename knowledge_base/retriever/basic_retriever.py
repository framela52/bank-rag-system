from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

class BasicRetriever:
    def __init__(self, store_name: str = "faiss_recursive"):
        print(f"Загрузка {store_name}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        store_path = Path("data/vector_stores") / store_name
        
        try:
            self.vector_store = FAISS.load_local(
                str(store_path),
                self.embeddings,  
                allow_dangerous_deserialization=True
            )
            print(f"Загружено: {self.vector_store.index.ntotal} векторов")
        except Exception as e:
            print(f"Ошибка загрузки FAISS: {e}")
            print("Пересоздаём...")
            self.rebuild_store(store_name)
    
    def rebuild_store(self, store_name: str):
        """Пересоздание FAISS если не загрузился"""
        import json
        chunks_file = Path("data/processed") / f"chunks_{store_name}.json"
        
        if chunks_file.exists():
            chunks = json.load(open(chunks_file, encoding='utf-8'))
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            
            self.vector_store = FAISS.from_texts(
                texts, self.embeddings, metadatas=metadatas
            )
            store_path = Path("data/vector_stores") / store_name
            self.vector_store.save_local(str(store_path))
            print(f"Пересоздано: {len(texts)} векторов")
    
    def similarity_search(self, query: str, k: int = 4):
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            print(" Нет векторного хранилища!")
            return []
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f" Ошибка поиска: {e}")
            return []
    
    def mmr_search(self, query: str, k: int = 4):
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            return []
        try:
            return self.vector_store.max_marginal_relevance_search(query, k=k)
        except Exception as e:
            print(f"MMR ошибка: {e}")
            return self.similarity_search(query, k=k)
    
    def compare_methods(self, query: str):
        print(f"\n{'='*50}")
        print(f"ТЕСТ: '{query}' ({self.vector_store.index.ntotal if hasattr(self, 'vector_store') else 0} векторов)")
        print('='*50)
        
        sim_docs = self.similarity_search(query, k=3)
        mmr_docs = self.mmr_search(query, k=3)
        
        print("\nSIMILARITY SEARCH:")
        for i, doc in enumerate(sim_docs, 1):
            print(f"{i}. [{doc.metadata.get('product_type', 'N/A')}] {len(doc.page_content)} симв.")
            print(f"   {doc.page_content[:120]}...")
        
        print("\nMMR SEARCH (разнообразие):")
        for i, doc in enumerate(mmr_docs, 1):
            print(f"{i}. [{doc.metadata.get('product_type', 'N/A')}] {len(doc.page_content)} симв.")
            print(f"   {doc.page_content[:120]}...")
        
        print('='*50)
