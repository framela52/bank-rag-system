from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from knowledge_base.chunking.strategies import ChunkingStrategies, compare_chunking_strategies
from knowledge_base.chunking.preprocess import load_and_clean_documents
from knowledge_base.embeddings.builder import EmbeddingBuilder
from knowledge_base.chunking.strategies import save_chunks

class ChunkingEvaluator:
    """Оценка качества разных стратегий чанкинга"""
    
    def __init__(self, embedding_model: str = "intfloat/multilingual-e5-large"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.test_queries = self._create_test_queries()
    
    def _create_test_queries(self) -> List[Dict]:
        """Тестовые запросы с эталонными ответами"""
        return [
            {
                "query": "Какие документы нужны для потребительского кредита?",
                "relevant_keywords": ["паспорт", "ИНН", "справка", "2-НДФЛ", "трудовая книжка"],
                "product_types": ["credit"]
            },
            {
                "query": "Какая ставка по депозиту 'Надежный выбор'?",
                "relevant_keywords": ["процент", "ставка", "надёжный выбор", "пополнение"],
                "product_types": ["deposit"]
            },
            {
                "query": "Условия ипотеки для семей с детьми",
                "relevant_keywords": ["ипотека", "семейная", "дети", "льготная", "взнос"],
                "product_types": ["mortgage"]
            },
            {
                "query": "Комиссия за переводы между своими счетами",
                "relevant_keywords": ["комиссия", "перевод", "тариф", "РКО"],
                "product_types": ["service"]
            },
            {
                "query": "Можно ли досрочно погасить кредит без штрафов?",
                "relevant_keywords": ["досрочное", "погашение", "штраф", "кредит"],
                "product_types": ["credit"]
            }
        ]
    
    def evaluate_chunking_quality(self, chunks: List[Dict], test_queries: List[Dict] = None) -> Dict:
        """Оценка качества чанков по семантической релевантности"""
        if test_queries is None:
            test_queries = self.test_queries
        
        results = []
        
        for query_info in test_queries:
            query_emb = self.embedding_model.encode([query_info["query"]])
            
            for chunk in chunks:
                chunk_emb = self.embedding_model.encode([chunk["text"]])
                similarity = cosine_similarity(query_emb, chunk_emb)[0][0]
                
                # Проверяем релевантность по ключевым словам
                keyword_match = self._keyword_relevance(chunk["text"], query_info["relevant_keywords"])
                
                results.append({
                    "query": query_info["query"],
                    "chunk_id": chunk["id"],
                    "similarity": float(similarity),
                    "keyword_score": keyword_match,
                    "total_score": float(similarity * keyword_match),
                    "product_match": query_info["product_types"] == [chunk["metadata"]["product_type"]],
                    "strategy": chunk["metadata"]["chunk_strategy"]
                })
        
        return self._aggregate_metrics(results)
    
    def _keyword_relevance(self, text: str, keywords: List[str]) -> float:
        """Оценка релевантности по ключевым словам"""
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return min(matches / len(keywords), 1.0)
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Агрегация метрик по запросам"""
        df = pd.DataFrame(results)
        
        metrics = {
            "mean_similarity": df["similarity"].mean(),
            "mean_keyword_score": df["keyword_score"].mean(),
            "mean_total_score": df["total_score"].mean(),
            "coverage_queries": len(df["query"].unique()),
            "product_accuracy": df["product_match"].mean(),
            "top_k_precision": self._top_k_precision(df, k=3),
            "strategy_performance": df.groupby("strategy")["total_score"].agg(['mean', 'count']).to_dict()
        }
        
        return metrics
    
    def _top_k_precision(self, df: pd.DataFrame, k: int = 3) -> float:
        """Precision@K для топ-k чанков по каждому запросу"""
        precisions = []
        for query in df["query"].unique():
            query_df = df[df["query"] == query].nlargest(k, "total_score")
            relevant = query_df[query_df["total_score"] > 0.3]  # Порог релевантности
            precision = len(relevant) / k if len(query_df) >= k else 0
            precisions.append(precision)
        return np.mean(precisions)

def run_comprehensive_evaluation():
    """Комплексная оценка всех стратегий чанкинга"""
    
    # Загружаем документы
    documents = load_and_clean_documents(Path("data/raw"))
    
    print("=== ТЕСТИРОВАНИЕ СТРАТЕГИЙ ЧАНКИНГА ===")
    
    # Создаем чанки для всех стратегий
    chunker = ChunkingStrategies(chunk_size=500, chunk_overlap=50)
    strategies = ["fixed_size", "by_sentences", "recursive", "markdown"]
    
    evaluator = ChunkingEvaluator()
    all_results = {}
    
    basic_stats = compare_chunking_strategies(documents, strategies)
    
    for strategy in strategies:
        print(f"\n--- Оценка стратегии: {strategy} ---")
        
        # Получаем чанки для стратегии
        if strategy == "fixed_size":
            chunks = chunker.by_fixed_size(documents)
        elif strategy == "by_sentences":
            chunks = chunker.by_sentences(documents)
        elif strategy == "recursive":
            chunks = chunker.recursive_split(documents)
        elif strategy == "markdown":
            chunks = chunker.by_markdown(documents)
        
        # Сохраняем чанки
        save_chunks(chunks, Path("data/processed"), strategy)
        
        # Оцениваем качество
        metrics = evaluator.evaluate_chunking_quality(chunks)
        all_results[strategy] = {
            **basic_stats[strategy],
            **metrics
        }
        
        print(f"Количество чанков: {len(chunks)}")
        print(f"Средний размер: {basic_stats[strategy]['avg_chunk_size']:.0f}")
        print(f"Средний total_score: {metrics['mean_total_score']:.3f}")
        print(f"Precision@3: {metrics['top_k_precision']:.3f}")
    
    # Создаем сравнительную таблицу
    comparison_df = pd.DataFrame({
    'Стратегия': list(all_results.keys()),
    'Чанки': [all_results[s]['num_chunks'] for s in all_results],
    'Размер': [int(all_results[s]['avg_chunk_size']) for s in all_results],
    'Total Score': [f"{all_results[s]['mean_total_score']:.3f}" for s in all_results],
    'Precision@3': [f"{all_results[s]['top_k_precision']:.3f}" for s in all_results],
    'Продуктовая точность': [f"{all_results[s]['product_accuracy']:.3f}" for s in all_results]
    })
    
    print("\n=== СРАВНИТЕЛЬНАЯ ТАБЛИЦА ===")
    print(comparison_df.to_string(index=False))
    
    # Сохраняем результаты
    comparison_df.to_csv("data/processed/chunking_comparison.csv", index=False)
    with open("data/processed/evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Рекомендация лучшей стратегии
    best_strategy = max(all_results.keys(), 
                       key=lambda s: all_results[s]['mean_total_score'] * all_results[s]['top_k_precision'])
    print(f"\n ЛУЧШАЯ СТРАТЕГИЯ: {best_strategy}")

def test_vector_retrieval_quality():
    """Тестирование качества векторного поиска"""
    from knowledge_base.embeddings.store import VectorStoreManager
    
    store_manager = VectorStoreManager()
    stores = ["faiss_recursive", "faiss_fixed_size", "faiss_markdown"]
    
    test_query = "Какие процентные ставки по депозиту на 6 месяцев?"
    
    results = store_manager.compare_stores(stores, test_query, k=3)
    
    print("\n=== ТЕСТ ВЕКТОРНОГО ПОИСКА ===")
    for store_name, docs in results.items():
        print(f"\n{store_name}:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. Score: {doc['score']:.4f}")
            print(f"   Продукт: {doc['metadata'].get('product_type', 'N/A')}")
            print(f"   Текст: {doc['text']}")

if __name__ == "__main__":
    run_comprehensive_evaluation()
    # test_vector_retrieval_quality()
