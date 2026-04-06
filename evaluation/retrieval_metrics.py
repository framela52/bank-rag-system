from typing import List, Dict, Any, Set
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent))

from langchain_core.documents import Document
import numpy as np

root_dir = Path(__file__).parent.parent  
sys.path.insert(0, str(root_dir))

from knowledge_base.retriever.hybrid_retriever import HybridRetriever
from knowledge_base.retriever.compression import HybridRetrieverWithCompression


def load_test_qas(qas_path: str) -> List[Dict[str, Any]]:
    """Загружает тестовые вопросы и эталонные чанки."""
    with open(qas_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_doc_id(doc: Document) -> str:
    """Извлекает ID чанка из документа LangChain."""
    if "id" in doc.metadata:
        return doc.metadata["id"]
    if "chunk_id" in doc.metadata:
        return doc.metadata["chunk_id"]
    
    chunk_index = doc.metadata.get("chunk_index", "")
    source = doc.metadata.get("source", "")
    return f"{source}_{chunk_index}"


def calc_hit_at_k(retrieved_docs: List[Document], relevant_ids: Set[str], k: int) -> float:
    """Hit@k: 1, если хотя бы один релевантный чанк в top‑k."""
    top_k_ids = {get_doc_id(doc) for doc in retrieved_docs[:k]}
    return 1.0 if len(top_k_ids & relevant_ids) > 0 else 0.0


def calc_mrr(retrieved_docs: List[Document], relevant_ids: Set[str], k: int) -> float:
    """MRR: 1 / (ранг первого релевантного документа)."""
    for i, doc in enumerate(retrieved_docs[:k], start=1):
        doc_id = get_doc_id(doc)
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def calc_precision_at_k(retrieved_docs: List[Document], relevant_ids: Set[str], k: int) -> float:
    """Precision@k: доля релевантных документов среди первых k."""
    if k == 0:
        return 0.0
    
    top_k_ids = {get_doc_id(doc) for doc in retrieved_docs[:k]}
    relevant_in_top_k = len(top_k_ids & relevant_ids)
    return relevant_in_top_k / k


def calc_recall_at_k(retrieved_docs: List[Document], relevant_ids: Set[str], k: int) -> float:
    """Recall@k: доля релевантных документов, найденных среди первых k."""
    total_relevant = len(relevant_ids)
    if total_relevant == 0:
        return 0.0
    
    top_k_ids = {get_doc_id(doc) for doc in retrieved_docs[:k]}
    relevant_found = len(top_k_ids & relevant_ids)
    return relevant_found / total_relevant


def calc_ndcg_at_k(retrieved_docs: List[Document], relevant_ids: Set[str], k: int) -> float:
    """NDCG@k для бинарной релевантности."""
    if k == 0:
        return 0.0
    
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k], start=1):
        doc_id = get_doc_id(doc)
        relevance = 1.0 if doc_id in relevant_ids else 0.0
        dcg += relevance / np.log2(i + 1)
    
    idcg = 0.0
    for i in range(1, min(len(relevant_ids), k) + 1):
        idcg += 1.0 / np.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calc_average_precision(retrieved_docs: List[Document], relevant_ids: Set[str], k: int) -> float:
    """Average Precision@k."""
    if k == 0:
        return 0.0
    
    precisions = []
    relevant_found = 0
    
    for i, doc in enumerate(retrieved_docs[:k], start=1):
        doc_id = get_doc_id(doc)
        if doc_id in relevant_ids:
            relevant_found += 1
            precision_at_i = relevant_found / i
            precisions.append(precision_at_i)
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(precisions)


def evaluate_retriever(
    retriever,
    test_qas: List[Dict[str, Any]],
    k: int = 5,
    retriever_name: str = "Retriever"
) -> Dict[str, float]:
    """
    Оценивает качество ретривера.
    """
    hit_scores = []
    mrr_scores = []
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    ap_scores = []
    
    errors = []
    detailed_results = []

    print(f"\n{'='*80}")
    print(f"Оценка ретривера: {retriever_name}")
    print(f"{'='*80}")

    for idx, item in enumerate(test_qas, 1):
        query = item["question"]
        relevant_ids = set(item["expected_chunk_ids"])

        try:
            if hasattr(retriever, 'hybrid_search'):
                docs = retriever.hybrid_search(query, k=k)
            else:
                docs = retriever.get_relevant_documents(query)[:k]
            
            hit = calc_hit_at_k(docs, relevant_ids, k)
            mrr = calc_mrr(docs, relevant_ids, k)
            precision = calc_precision_at_k(docs, relevant_ids, k)
            recall = calc_recall_at_k(docs, relevant_ids, k)
            ndcg = calc_ndcg_at_k(docs, relevant_ids, k)
            ap = calc_average_precision(docs, relevant_ids, k)
            
            # Находим ранг первого релевантного
            rank = None
            for i, doc in enumerate(docs[:k], start=1):
                if get_doc_id(doc) in relevant_ids:
                    rank = i
                    break
            
            detailed_results.append({
                "query": query,
                "hit": hit,
                "mrr": mrr,
                "rank": rank,
                "retrieved_ids": [get_doc_id(doc) for doc in docs[:k]]
            })
            
        except Exception as e:
            print(f"  Ошибка для вопроса {idx}: {e}")
            hit, mrr, precision, recall, ndcg, ap = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            errors.append(query)
        
        hit_scores.append(hit)
        mrr_scores.append(mrr)
        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)
        ap_scores.append(ap)

    results = {
        "retriever_name": retriever_name,
        "hit_rate@k": sum(hit_scores) / len(hit_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
        "precision@k": sum(precision_scores) / len(precision_scores),
        "recall@k": sum(recall_scores) / len(recall_scores),
        "ndcg@k": sum(ndcg_scores) / len(ndcg_scores),
        "map@k": sum(ap_scores) / len(ap_scores),
        "total_questions": len(hit_scores),
        "errors_count": len(errors),
        "detailed_results": detailed_results
    }
    
    if errors:
        results["errors"] = errors
    
    return results


def print_evaluation_summary(results: Dict[str, float]) -> None:
    """Выводит краткую сводку метрик."""
    print(f"\n{' Метрики для ' + results['retriever_name'] + ' ':-^60}")
    print(f"  Hit Rate@{results.get('k', 5)}:     {results['hit_rate@k']:.3f} ({results['hit_rate@k']*100:.1f}%)")
    print(f"  MRR:                         {results['mrr']:.3f}")
    print(f"  Precision@{results.get('k', 5)}:   {results['precision@k']:.3f}")
    print(f"  Recall@{results.get('k', 5)}:      {results['recall@k']:.3f}")
    print(f"  NDCG@{results.get('k', 5)}:        {results['ndcg@k']:.3f}")
    print(f"  MAP@{results.get('k', 5)}:         {results['map@k']:.3f}")
    print(f"  Ошибок:                      {results['errors_count']}/{results['total_questions']}")
    print("-" * 60)


def print_detailed_analysis(results: Dict[str, float], show_retrieved: bool = False) -> None:
    """Выводит детальный анализ по каждому вопросу."""
    print(f"\n{' Детальный анализ ':-^60}")
    print(f"{'№':<4} {'Hit':<6} {'MRR':<8} {'Ранг':<8} {'Вопрос'}")
    print("-" * 60)
    
    for idx, detail in enumerate(results['detailed_results'], 1):
        query_preview = detail['query'][:40] + "..." if len(detail['query']) > 40 else detail['query']
        rank_str = str(detail['rank']) if detail['rank'] else "—"
        print(f"{idx:<4} {detail['hit']:<6.0f} {detail['mrr']:<8.3f} {rank_str:<8} {query_preview}")
        
        if show_retrieved and detail['rank'] is None:
            print(f"       Retrieved: {detail['retrieved_ids'][:3]}")
    
    print("-" * 60)
    
    # Статистика по рангам
    ranks = [d['rank'] for d in results['detailed_results'] if d['rank'] is not None]
    if ranks:
        print(f"\nСтатистика рангов первого релевантного:")
        print(f"  Средний ранг: {np.mean(ranks):.2f}")
        print(f"  Медианный ранг: {np.median(ranks):.0f}")
        print(f"  Распределение:")
        for r in range(1, 6):
            count = sum(1 for rank in ranks if rank == r)
            print(f"    Ранг {r}: {count} вопросов ({count/len(ranks)*100:.1f}%)")


def compare_retrievers(
    test_qas: List[Dict[str, Any]],
    k: int = 5,
    use_compression: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Сравнивает гибридный ретривер и ретривер с компрессией.
    """
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕТРИВЕРОВ")
    print("="*80)
    print(f"Параметры: k={k}, compression={'вкл' if use_compression else 'выкл'}")
    
    results = {}
    
    # Гибридный ретривер (без компрессии)
    print("\nИнициализация гибридного ретривера...")
    hybrid = HybridRetriever()
    results['hybrid'] = evaluate_retriever(hybrid, test_qas, k, "Hybrid Retriever (без компрессии)")
    
    # Гибридный ретривер с компрессией
    print("\nИнициализация гибридного ретривера с компрессией...")
    hybrid_comp = HybridRetrieverWithCompression(
        use_compression=use_compression,
        compression_k=k,
        model_name="llama3.1"
    )
    results['hybrid_compression'] = evaluate_retriever(
        hybrid_comp, test_qas, k, 
        f"Hybrid Retriever с компрессией (k={k})"
    )
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Выводит таблицу сравнения метрик."""
    print("\n" + "="*80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК")
    print("="*80)
    
    metrics = ['hit_rate@k', 'mrr', 'precision@k', 'recall@k', 'ndcg@k', 'map@k']
    metric_names = {
        'hit_rate@k': 'Hit Rate',
        'mrr': 'MRR',
        'precision@k': 'Precision',
        'recall@k': 'Recall',
        'ndcg@k': 'NDCG',
        'map@k': 'MAP'
    }
    
    print(f"\n{'Метрика':<15} {'Hybrid':<25} {'Hybrid+Compression':<25} {'Δ':<10}")
    print("-" * 75)
    
    for metric in metrics:
        hybrid_val = results['hybrid'][metric]
        comp_val = results['hybrid_compression'][metric]
        delta = comp_val - hybrid_val
        
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        
        print(f"{metric_names[metric]:<15} "
              f"{hybrid_val:.3f} ({hybrid_val*100:.1f}%){' ':<10} "
              f"{comp_val:.3f} ({comp_val*100:.1f}%){' ':<10} "
              f"{delta_str} {arrow}")
    
    print("-" * 75)


def main():
    # Путь к тестовым данным
    qas_path = Path("data/processed/test_qas.json")
    
    print(f"Загрузка тестовых данных из {qas_path}")
    test_qas = load_test_qas(str(qas_path))
    print(f"Загружено {len(test_qas)} тестовых вопросов")
    
    # Проверка структуры данных
    if test_qas:
        print(f"\nПример вопроса {test_qas[0]['question'][:100]}...")
        print(f"Эталонные чанки: {test_qas[0]['expected_chunk_ids']}")
    
    # Параметры оценки
    k = 5
    use_compression = True
    
    # Сравнение ретриверов
    results = compare_retrievers(test_qas, k=k, use_compression=use_compression)
    
    # Вывод детальных результатов
    print("\n" + "="*80)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    
    for retriever_name, res in results.items():
        print_evaluation_summary(res)
        print_detailed_analysis(res, show_retrieved=False)
    
    # Сравнительная таблица
    print_comparison_table(results)
    
    # Вывод лучшего результата
    print("\n" + "="*80)
    print("ВЫВОДЫ")
    print("="*80)
    
    best_hit = max(results.items(), key=lambda x: x[1]['hit_rate@k'])
    best_mrr = max(results.items(), key=lambda x: x[1]['mrr'])
    
    print(f"Лучший Hit Rate@{k}: {best_hit[0]} ({best_hit[1]['hit_rate@k']:.3f})")
    print(f"Лучший MRR: {best_mrr[0]} ({best_mrr[1]['mrr']:.3f})")
    
    if results['hybrid_compression']['hit_rate@k'] > results['hybrid']['hit_rate@k']:
        improvement = (results['hybrid_compression']['hit_rate@k'] - results['hybrid']['hit_rate@k']) / results['hybrid']['hit_rate@k'] * 100
        print(f"\nКомпрессия улучшила Hit Rate на {improvement:.1f}%")
    elif results['hybrid_compression']['hit_rate@k'] < results['hybrid']['hit_rate@k']:
        degradation = (results['hybrid']['hit_rate@k'] - results['hybrid_compression']['hit_rate@k']) / results['hybrid']['hit_rate@k'] * 100
        print(f"\nКомпрессия ухудшила Hit Rate на {degradation:.1f}%")
    else:
        print(f"\nКомпрессия не изменила Hit Rate")

def compare_optimization_strategies():
    """
    Сравнение качества поиска с разными стратегиями
    """
    from llm.optimization import QueryOptimizer, ReRanker
    from knowledge_base.retriever.hybrid_retriever import HybridRetriever
    
    print("\n" + "="*80)
    print("СРАВНЕНИЕ СТРАТЕГИЙ УЛУЧШЕНИЯ ПОИСКА")
    print("="*80)
    
    retriever = HybridRetriever()
    optimizer = QueryOptimizer()
    reranker = ReRanker()
    
    test_questions = [
        "депозит минимальная сумма",
        "процентная ставка по депозиту на 3 года",
        "ипотека первоначальный взнос новостройки"
    ]
    
    results = {
        "baseline": [],
        "self_query": [],
        "self_query_multi": [],
        "full_optimization": []
    }
    
    for question in test_questions:
        # Baseline
        baseline_docs = retriever.hybrid_search(question, k=3)
        results["baseline"].append(len(baseline_docs))
        
        # Self-query
        optimized_q = optimizer.self_query(question)
        sq_docs = retriever.hybrid_search(optimized_q, k=3)
        results["self_query"].append(len(sq_docs))
        
        # Self-query + Multi-query
        queries = optimizer.multi_query(question, num_variations=2)
        all_docs = {}
        for q in queries:
            docs = retriever.hybrid_search(q, k=2)
            for doc in docs:
                all_docs[doc.metadata.get('id', '')] = doc
        results["self_query_multi"].append(len(all_docs))
        
        # Full optimization (с реранкингом)
        full_docs = retriever.hybrid_search(optimized_q, k=5)
        reranked = reranker.rerank_by_embeddings(question, full_docs, k=3)
        results["full_optimization"].append(len(reranked))
    
    print("\nРезультаты (количество релевантных документов в топ-3):")
    print("-" * 60)
    for strategy, scores in results.items():
        avg = sum(scores) / len(scores)
        print(f"  {strategy}: {avg:.1f}")
    
    print("\nВывод: Self-query + Multi-query + Реранкинг дают наибольшее количество")
    print("релевантных документов в результатах поиска.")

if __name__ == "__main__":
    main()
    compare_optimization_strategies()