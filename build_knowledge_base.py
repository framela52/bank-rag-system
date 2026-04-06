from pathlib import Path
from knowledge_base.chunking.preprocess import load_and_clean_documents
from knowledge_base.chunking.strategies import compare_chunking_strategies, save_chunks
from knowledge_base.embeddings.store import VectorStoreManager
from evaluation.chunking_eval import run_comprehensive_evaluation
import pandas as pd

def main():
    raw_data_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    vector_dir = Path("data/vector_stores")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== 1. ГЛУБОКАЯ ОЧИСТКА ДОКУМЕНТОВ ===")
    documents = load_and_clean_documents(raw_data_dir)
    
    print("\n=== 2. СРАВНЕНИЕ ЧАНКИНГА ===")
    basic_results = compare_chunking_strategies(documents)
    
    print("\n=== 3. КАЧЕСТВЕННАЯ ОЦЕНКА ===")
    
    run_comprehensive_evaluation()
    
    comparison_path = Path("data/processed/chunking_comparison.csv")
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        df['Total_Score'] = pd.to_numeric(df['Total Score'], errors='coerce')
        best_idx = df['Total_Score'].idxmax()
        best_strategy = df.loc[best_idx, 'Стратегия']
        print(f"По метрике Total Score: {best_strategy}")
    else:
        best_strategy = "by_sentences"  
        print(f" Фиксируем: {best_strategy}")

    print(f"\nИСПОЛЬЗУЕМ ЛУЧШУЮ: {best_strategy}")
    
    print("\n=== 4. ЭМБЕДДИНГИ + FAISS ===")
    store_manager = VectorStoreManager(base_path=vector_dir)
    chunks_file = processed_dir / f"chunks_{best_strategy}.json"
    
    if chunks_file.exists():
        success = store_manager.build_from_chunks(chunks_file, f"faiss_{best_strategy}")
        if success:
            print(" ВЕКТОРНАЯ БД СОЗДАНА!")
    
    print("\n=== 5. ТЕСТ ПОИСКА ===")
    test_query = "Какие документы нужны для кредита?"
    store_name = f"faiss_{best_strategy}"
    store = store_manager.get_store(store_name)
    
    if store:
        docs = store.similarity_search(test_query, k=3)
        print(f"\n ТОП-3 для '{test_query}':")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.metadata.get('product_type', 'N/A')}")
            print(f"   {doc.page_content[:100]}...\n")
    
        print(" data/processed/chunking_comparison.csv")
    print(" data/vector_stores/faiss_recursive/")

if __name__ == "__main__":
    main()
