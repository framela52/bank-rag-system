import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.manual_eval import ManualRAGEvaluator
from llm.optimized_chain import test_performance


def main():
    print("="*80)
    print("ПОЛНАЯ ОЦЕНКА RAG СИСТЕМЫ (РУЧНЫЕ МЕТРИКИ)")
    print("="*80)
    
    # Оценка качества ручными метриками
    print("\n1. ОЦЕНКА КАЧЕСТВА (Manual Metrics)")
    print("-" * 40)
    evaluator = ManualRAGEvaluator()
    results = evaluator.run_full_evaluation()
    evaluator.print_summary(results)
    
    # Тест производительности
    print("\n2. ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("-" * 40)
    test_performance()
    
    # Анализ и рекомендации
    print("\n3. АНАЛИЗ И РЕКОМЕНДАЦИИ")
    print("-" * 40)
    
    scores = results["manual_scores"]
    
    recommendations = []
    
    if scores["faithfulness"] < 0.7:
        recommendations.append(" Faithfulness: улучшить промпт, добавить верификацию фактов")
    
    if scores["answer_relevancy"] < 0.7:
        recommendations.append(" Answer Relevancy: оптимизировать ретривер, улучшить чанки")
    
    if scores["context_relevancy"] < 0.7:
        recommendations.append(" Context Relevancy: увеличить k, добавить реранкинг")
    
    if results["performance"]["average_response_time"] > 10:
        recommendations.append("Производительность: ускорить LLM, добавить кэш")
    
    if recommendations:
        print("Рекомендации:")
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("Система работает отлично!")


if __name__ == "__main__":
    main()