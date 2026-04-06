from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.chain import GigaChatClient, format_docs, extract_sources
from knowledge_base.retriever.compression import HybridRetrieverWithCompression
from config.settings import settings


class QueryOptimizer:
    """Улучшение запросов перед поиском"""
    
    def __init__(self):
        self.client = GigaChatClient()
    
    def self_query(self, original_question: str) -> str:
        """
        Self-query: улучшает запрос для более точного поиска
        """
        prompt = f"""Ты - помощник, который улучшает поисковые запросы для банковской документации.
Преобразуй вопрос пользователя в более точный запрос, выделяя ключевые термины.

Правила:
1. Удали лишние слова (приветствия, вежливости)
2. Выдели ключевые термины (сумма, ставка, срок, документы)
3. Если вопрос о конкретном продукте, укажи его тип (депозит, кредит, ипотека)
4. Сохрани числовые значения

Оригинальный вопрос: {original_question}

Улучшенный запрос:"""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100
            )
            return response.strip()
        except Exception as e:
            print(f"Self-query ошибка: {e}")
            return original_question
    
    def multi_query(self, original_question: str, num_variations: int = 3) -> List[str]:
        """
        Multi-query: генерирует альтернативные формулировки вопроса
        """
        prompt = f"""Ты - помощник, который генерирует альтернативные формулировки вопросов для банковского поиска.
Сгенерируй {num_variations} разных вариантов вопроса, которые помогут найти ту же информацию.

Оригинальный вопрос: {original_question}

Правила:
1. Используй разные формулировки и синонимы
2. Меняй порядок слов
3. Один вариант сделай более коротким, другой - более подробным
4. Не меняй смысл вопроса

Варианты вопросов (каждый на новой строке, без номеров):"""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200
            )
            variations = [v.strip() for v in response.split('\n') if v.strip()]
            # Убираем возможные номера в начале
            variations = [v.lstrip('1234567890.-) ') for v in variations]
            # Возвращаем оригинал + вариации
            return [original_question] + variations[:num_variations]
        except Exception as e:
            print(f"Multi-query ошибка: {e}")
            return [original_question]


class ReRanker:
    """Реранкинг результатов поиска"""
    
    def __init__(self):
        self.client = GigaChatClient()
    
    def rerank_by_llm(self, query: str, documents: List, k: int = 5) -> List:
        """
        Реранкинг через LLM: оценивает релевантность каждого документа
        """
        if not documents:
            return []
        
        # Ограничиваем количество для реранкинга (чтобы не было слишком долго)
        docs_to_rerank = documents[:10]
        
        scored_docs = []
        
        for doc in docs_to_rerank:
            prompt = f"""Оцени релевантность документа к вопросу по шкале от 0 до 10.
Дай ТОЛЬКО число, без пояснений.

Вопрос: {query}

Документ:
{doc.page_content[:500]}

Оценка релевантности (только число от 0 до 10):"""
            
            try:
                response = self.client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10
                )
                # Извлекаем число из ответа
                import re
                numbers = re.findall(r'\d+', response)
                score = int(numbers[0]) / 10 if numbers else 0.5
            except Exception:
                score = 0.5
            
            scored_docs.append((doc, score))
        
        # Сортируем по убыванию оценки
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:k]]
    
    def rerank_by_embeddings(self, query: str, documents: List, k: int = 5) -> List:
        """
        Реранкинг через эмбеддинги (быстрее, чем LLM)
        """
        if not documents:
            return []
        
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        query_emb = model.encode([query])[0]
        doc_embs = model.encode([doc.page_content for doc in documents[:20]])
        
        # Вычисляем косинусное сходство
        similarities = np.dot(doc_embs, query_emb) / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Сортируем документы по сходству
        scored = list(zip(documents[:20], similarities))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored[:k]]
    
    def hybrid_rerank(self, query: str, documents: List, k: int = 5) -> List:
        """
        Гибридный реранкинг: эмбеддинги (быстро) + LLM (точно) для топ-3
        """
        if not documents:
            return []
        
        # Сначала быстрый реранкинг через эмбеддинги
        emb_reranked = self.rerank_by_embeddings(query, documents, k=10)
        
        # Затем LLM реранкинг для топ-5 для повышения точности
        if len(emb_reranked) > k:
            llm_reranked = self.rerank_by_llm(query, emb_reranked[:k*2], k=k)
            return llm_reranked
        
        return emb_reranked[:k]


class OptimizedRAGWithStrategies:
    """
    RAG цепочка с оптимизациями:
    - Self-query
    - Multi-query  
    - Реранкинг
    """
    
    def __init__(self, use_compression: bool = True):
        print("Инициализация RAG с расширенными стратегиями...")
        
        self.client = GigaChatClient()
        self.retriever = HybridRetrieverWithCompression(
            use_compression=use_compression,
            compression_k=settings.RETRIEVAL_K
        )
        self.query_optimizer = QueryOptimizer()
        self.reranker = ReRanker()
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        
        print("RAG с оптимизациями готова!")
    
    def search_with_strategies(self, question: str, k: int = 5, 
                                use_self_query: bool = True,
                                use_multi_query: bool = True,
                                use_rerank: bool = True) -> Tuple[List, Dict]:
        """
        Поиск с применением всех стратегий
        """
        print(f"\nПоиск для: {question[:50]}...")
        
        # 1. Self-query: улучшаем запрос
        if use_self_query:
            optimized_query = self.query_optimizer.self_query(question)
            print(f"   Self-query: {optimized_query}")
        else:
            optimized_query = question
        
        # 2. Multi-query: генерируем альтернативные запросы
        if use_multi_query:
            queries = self.query_optimizer.multi_query(optimized_query, num_variations=3)
            print(f"   Multi-query: {len(queries)} вариантов")
        else:
            queries = [optimized_query]
        
        # 3. Поиск по всем вариантам запросов
        all_docs = {}
        for q in queries:
            docs = self.retriever.hybrid_search(q, k=k*2)
            for doc in docs:
                doc_id = doc.metadata.get('id', '')
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc
        
        docs = list(all_docs.values())[:k*3]
        print(f"   Найдено уникальных документов: {len(docs)}")
        
        # 4. Реранкинг результатов
        if use_rerank and docs:
            print(f"   Реранкинг через эмбеддинги...")
            docs = self.reranker.rerank_by_embeddings(question, docs, k=k)
        
        return docs, {"optimized_query": optimized_query, "num_queries": len(queries)}
    
    def ask(self, question: str, k: int = None,
            use_self_query: bool = True,
            use_multi_query: bool = True,
            use_rerank: bool = True) -> Dict[str, Any]:
        """
        Задаёт вопрос с применением всех стратегий
        """
        if k is None:
            k = settings.RETRIEVAL_K
        
        # Поиск с оптимизациями
        docs, search_info = self.search_with_strategies(
            question, k, use_self_query, use_multi_query, use_rerank
        )
        
        if not docs:
            return {
                "answer": "Извините, я не могу найти информацию по вашему вопросу.",
                "sources": [],
                "context_used": False,
                "search_info": search_info
            }
        
        # Форматируем контекст
        context = format_docs(docs)
        
        # Системный промпт
        system_prompt = f"""Ты - профессиональный банковский консультант. Отвечай на вопросы клиента, используя ТОЛЬКО информацию из документов ниже.

Документы:
{context}

Правила:
1. Отвечай только на основе предоставленных документов.
2. Не используй эмодзи.
3. Если информации нет, скажи об этом честно.
4. Приводи конкретные цифры из документов.
5. Указывай источник информации.

Вопрос клиента: {question}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        print("Генерация ответа через GigaChat...")
        
        try:
            answer = self.client.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            sources = extract_sources(docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": True,
                "num_docs": len(docs),
                "search_info": search_info
            }
            
        except Exception as e:
            return {
                "answer": f"Ошибка: {e}",
                "sources": [],
                "context_used": False,
                "search_info": search_info
            }


def compare_strategies():
    """
    Сравнение качества с разными стратегиями
    """
    print("="*80)
    print("СРАВНЕНИЕ СТРАТЕГИЙ УЛУЧШЕНИЯ КАЧЕСТВА")
    print("="*80)
    
    test_questions = [
        "Какая минимальная сумма для открытия депозита?",
        "Можно ли досрочно погасить кредит?",
        "Документы для ипотеки"
    ]
    
    rag = OptimizedRAGWithStrategies(use_compression=True)
    
    configs = [
        ("Без оптимизаций", False, False, False),
        ("Только self-query", True, False, False),
        ("Self-query + multi-query", True, True, False),
        ("Self-query + multi-query + реранкинг", True, True, True),
    ]
    
    for name, use_sq, use_mq, use_rr in configs:
        print(f"\n{'='*60}")
        print(f"Конфигурация: {name}")
        print('='*60)
        
        for question in test_questions:
            print(f"\nВопрос: {question}")
            result = rag.ask(question, use_self_query=use_sq, 
                            use_multi_query=use_mq, use_rerank=use_rr)
            print(f"Ответ: {result['answer'][:200]}...")
            print(f"Найдено документов: {result['num_docs']}")
            if result.get('search_info'):
                print(f"Оптимизированный запрос: {result['search_info'].get('optimized_query', 'N/A')}")


if __name__ == "__main__":
    compare_strategies()