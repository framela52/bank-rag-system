import time
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.optimization import OptimizedRAGWithStrategies, compare_strategies
from llm.chain import RAGChain
from llm.optimization import QueryOptimizer, ReRanker
from config.settings import settings

class OptimizedRAGChainWithStrategies(OptimizedRAGWithStrategies):
    """
    Расширенная версия с кэшированием + всеми стратегиями
    """
    
    def __init__(self, use_compression: bool = True, cache_ttl: int = 3600):
        super().__init__(use_compression)
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.request_times = []
    
    def get_cache_stats(self):
        """Статистика кэша для тестовой функции"""
        try:
            if hasattr(self.llm_chain.llm, 'cache') and self.llm_chain.llm.cache:
                cache = self.llm_chain.llm.cache
                hits = len([k for k, v in cache.items() if v['hit']])
                total = len(cache)
                return {
                    'hits': hits,
                    'total': total,
                    'hit_rate': hits / total if total > 0 else 0.0
                }
            return {'hits': 0, 'total': 0, 'hit_rate': 0.0}
        except:
            return {'hits': 0, 'total': 0, 'hit_rate': 0.0}
    
    def ask_with_cache(self, question: str, k: int = None,
                       use_self_query: bool = True,
                       use_multi_query: bool = True,
                       use_rerank: bool = True) -> Dict[str, Any]:
        """
        Задаёт вопрос с кэшированием и всеми стратегиями
        """
        import hashlib
        from datetime import datetime, timedelta
        
        if k is None:
            k = settings.RETRIEVAL_K
        
        # Ключ кэша учитывает все параметры
        cache_key = hashlib.md5(
            f"{question}_{k}_{use_self_query}_{use_multi_query}_{use_rerank}".encode()
        ).hexdigest()
        
        # Проверяем кэш
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() - entry["timestamp"] < timedelta(seconds=self.cache_ttl):
                print("Использован кэшированный ответ")
                return entry["result"]
        
        # Выполняем запрос
        start_time = time.time()
        result = self.ask(question, k, use_self_query, use_multi_query, use_rerank)
        response_time = time.time() - start_time
        
        result["response_time"] = response_time
        
        # Сохраняем в кэш
        self.cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now()
        }
        
        self.request_times.append(response_time)
        
        return result

def test_performance():
    """Сравнение производительности до и после оптимизации"""
    print("="*80)
    print("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*80)
    
    from llm.chain import RAGChain
    
    test_questions = [
        "Какая минимальная сумма для открытия депозита?",
        "Можно ли досрочно погасить кредит без штрафов?",
        "Какой первоначальный взнос по ипотеке на новостройки?",
        "Какие документы нужны для потребительского кредита?"
    ]
    
    # Тест без оптимизации
    print("\n1. Тест без оптимизации (обычный RAG)...")
    rag_standard = RAGChain(use_compression=True)
    
    standard_times = []
    for question in test_questions:
        start = time.time()
        rag_standard.ask(question)
        standard_times.append(time.time() - start)
    
    avg_standard = sum(standard_times) / len(standard_times)
    print(f"   Среднее время: {avg_standard:.2f} сек")
    
    # Тест с оптимизацией (первый запуск - заполнение кэша)
    print("\n2. Тест с оптимизацией (первый запуск)...")
    rag_optimized = OptimizedRAGChainWithStrategies(use_compression=True, cache_ttl=3600)
    
    optimized_times = []
    for question in test_questions:
        start = time.time()
        rag_optimized.ask_with_cache(question, use_rerank=False)
        optimized_times.append(time.time() - start)
    
    avg_optimized_first = sum(optimized_times) / len(optimized_times)
    print(f"   Среднее время (первый запуск): {avg_optimized_first:.2f} сек")
    
    # Тест с кэшем (второй запуск)
    print("\n3. Тест с оптимизацией (из кэша)...")
    cached_times = []
    for question in test_questions:
        start = time.time()
        rag_optimized.ask_with_cache(question, use_rerank=False)
        cached_times.append(time.time() - start)
    
    avg_cached = sum(cached_times) / len(cached_times)
    print(f"   Среднее время (из кэша): {avg_cached:.2f} сек")
    
    # Сравнение
    print("\n" + "="*80)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*80)
    print(f"Без оптимизации: {avg_standard:.2f} сек")
    print(f"С оптимизацией (первый запрос): {avg_optimized_first:.2f} сек")
    print(f"С оптимизацией (из кэша): {avg_cached:.2f} сек")
    
    if avg_cached > 0:
        print(f"Ускорение с кэшем: {avg_standard/avg_cached:.1f}x")
    
    # Статистика кэша
    print("\nСтатистика кэша:")
    stats = rag_optimized.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")



def test_all_optimizations():
    """
    Полное тестирование всех оптимизаций
    """
    print("="*80)
    print("ТЕСТИРОВАНИЕ ВСЕХ СТРАТЕГИЙ УЛУЧШЕНИЯ")
    print("="*80)
    
    # 1. Тест Self-query
    print("\n1. Self-query:")
    optimizer = QueryOptimizer()
    test_q = "скажите пожалуйста какая минимальная сумма для открытия депозита в вашем банке?"
    improved = optimizer.self_query(test_q)
    print(f"   Оригинал: {test_q}")
    print(f"   Улучшенный: {improved}")
    
    # 2. Тест Multi-query
    print("\n2. Multi-query:")
    variations = optimizer.multi_query("какие документы нужны для кредита?")
    for i, v in enumerate(variations, 1):
        print(f"   {i}. {v}")
    
    # 3. Сравнение стратегий
    print("\n3. Сравнение качества с разными стратегиями:")
    compare_strategies()


if __name__ == "__main__":
    test_all_optimizations()