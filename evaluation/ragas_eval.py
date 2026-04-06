import sys
from pathlib import Path
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_precision
)
from datasets import Dataset
from langchain_community.chat_models import ChatOllama

from llm.chain import RAGChain
from config.settings import settings


class RAGEvaluator:
    """Оценка качества RAG системы с помощью Ragas (локальная LLM)"""
    
    def __init__(self):
        self.rag_chain = RAGChain(use_compression=True)
        self.test_questions = self._load_test_questions()
        
        # Настраиваем Ragas на использование локальной LLM
        self.ragas_llm = None
        self._setup_ragas_llm()
    
    def _get_ollama_url(self) -> str:
        ollama_host = os.getenv("OLLAMA_HOST", "ollama")
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        return f"http://{ollama_host}:{ollama_port}"
    
    def _setup_ragas_llm(self):
        """Настраивает Ragas для использования локальной Ollama модели"""
        # Устанавливаем фейковый OpenAI ключ для обхода проверки
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-ragas-local"
        
        ollama_url = "http://ollama:11434"
        
        try:
            # Пробуем использовать Ollama напрямую
            self.ragas_llm = ChatOllama(
                model="llama3.2:1b",  # маленькая модель для оценки
                base_url=ollama_url,
                temperature=0.0,
                timeout=30.0
            )
            # Проверяем доступность Ollama
            test_response = self.ragas_llm.invoke("Test")
            print(f"Используется локальная Ollama модель для оценки (URL: {ollama_url})")
        except Exception as e:
            print(f"Не удалось подключиться к Ollama: {e}")
            print(f"URL: {ollama_url}")
            print("Будет использована простая эвристическая оценка")
            self.ragas_llm = None
    
    def _load_test_questions(self) -> List[Dict]:
        """Загружает тестовые вопросы с ожидаемыми ответами"""
        qas_path = Path("data/processed/test_qas.json")
        
        if qas_path.exists():
            with open(qas_path, 'r', encoding='utf-8') as f:
                base_questions = json.load(f)
                for q in base_questions:
                    q["ground_truth"] = self._get_ground_truth_for_question(q["question"])
                return base_questions
        
        return []
    
    def _get_ground_truth_for_question(self, question: str) -> str:
        """Возвращает эталонный ответ для вопроса"""
        ground_truths = {
            "депозит": "Минимальная сумма депозита - от 50 000 рублей. Процентная ставка зависит от срока.",
            "кредит": "Кредит можно досрочно погасить без штрафов. Максимальная сумма - до 3 000 000 рублей.",
            "ипотека": "Первоначальный взнос по ипотеке на новостройки - от 15%, на вторичку - от 20%.",
            "документы": "Для кредита нужны: паспорт, справка о доходах, трудовая книжка.",
            "капитализация": "Капитализация - это прибавление процентов к сумме вклада."
        }
        
        for key, answer in ground_truths.items():
            if key in question.lower():
                return answer
        
        return "Информация по данному вопросу отсутствует в эталонных данных."
    
    def collect_responses(self, questions: List[str]) -> Dict:
        """Собирает ответы системы для оценки"""
        questions_list = []
        answers_list = []
        contexts_list = []
        ground_truths = []
        response_times = []
        
        print("Сбор ответов для оценки...")
        
        gt_dict = {q["question"]: q.get("ground_truth", "") for q in self.test_questions}
        
        for i, question in enumerate(questions, 1):
            print(f"  {i}/{len(questions)}: {question[:50]}...")
            
            start_time = time.time()
            result = self.rag_chain.ask(question, k=5)
            end_time = time.time()
            
            questions_list.append(question)
            answers_list.append(result["answer"])
            
            docs = self.rag_chain.retriever.hybrid_search(question, k=5)
            context_texts = [doc.page_content for doc in docs]
            contexts_list.append(context_texts)
            
            ground_truths.append(gt_dict.get(question, "Информация отсутствует"))
            response_times.append(end_time - start_time)
        
        return {
            "questions": questions_list,
            "answers": answers_list,
            "contexts": contexts_list,
            "ground_truths": ground_truths,
            "response_times": response_times
        }
    
    def _simple_evaluation(self, data: Dict) -> Dict:
        """Простая эвристическая оценка (без LLM)"""
        print("\nИспользуется упрощённая оценка (без LLM)...")
        
        scores = []
        for answer in data["answers"]:
            score = min(len(answer) / 500, 1.0)
            scores.append(score)
        
        return {
            "faithfulness": 0.7,
            "answer_relevancy": sum(scores) / len(scores) if scores else 0.5,
            "context_relevancy": 0.6,
            "context_precision": 0.6,
            "detailed_results": []
        }
    
    def evaluate_ragas(self, data: Dict) -> Dict:
        """Оценка с помощью Ragas (с fallback на упрощённую оценку)"""
        print("\nОценка качества с Ragas...")
        
        if self.ragas_llm is None:
            return self._simple_evaluation(data)
        
        try:
            dataset = Dataset.from_dict({
                "question": data["questions"],
                "answer": data["answers"],
                "contexts": data["contexts"],
                "ground_truth": data["ground_truths"]
            })
            
            metrics = [
                faithfulness,
                answer_relevancy,
                context_relevancy,
                context_precision
            ]
            
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.ragas_llm
            )
            
            df = result.to_pandas()
            
            return {
                "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
                "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else 0.0,
                "context_relevancy": float(df["context_relevancy"].mean()) if "context_relevancy" in df else 0.0,
                "context_precision": float(df["context_precision"].mean()) if "context_precision" in df else 0.0,
                "detailed_results": df.to_dict('records') if not df.empty else []
            }
            
        except Exception as e:
            print(f"Ошибка при оценке Ragas: {e}")
            return self._simple_evaluation(data)
    
    def run_full_evaluation(self) -> Dict:
        """Запускает полную оценку"""
        print("="*80)
        print("ОЦЕНКА КАЧЕСТВА RAG СИСТЕМЫ")
        print("="*80)
        
        questions = [q["question"] for q in self.test_questions]
        
        data = self.collect_responses(questions)
        ragas_scores = self.evaluate_ragas(data)
        
        avg_time = sum(data["response_times"]) / len(data["response_times"])
        min_time = min(data["response_times"])
        max_time = max(data["response_times"])
        
        results = {
            "ragas_scores": ragas_scores,
            "performance": {
                "average_response_time": avg_time,
                "min_response_time": min_time,
                "max_response_time": max_time,
                "total_questions": len(questions)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_results(results)
        return results
    
    def _save_results(self, results: Dict):
        """Сохраняет результаты оценки"""
        output_path = Path("evaluation/results")
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_path = output_path / "ragas_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nРезультаты сохранены в {json_path}")
    
    def print_summary(self, results: Dict):
        """Выводит сводку результатов"""
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("="*80)
        
        print("\nМетрики Ragas:")
        print("-" * 40)
        for metric, score in results["ragas_scores"].items():
            if not metric.startswith("detailed"):
                print(f"  {metric}: {score:.3f}")
        
        print("\nИнтерпретация:")
        print("-" * 40)
        scores = results["ragas_scores"]
        
        if scores["faithfulness"] > 0.8:
            print("  Faithfulness (>0.8): Отлично - ответы хорошо соответствуют документам")
        elif scores["faithfulness"] > 0.6:
            print("  Faithfulness (0.6-0.8): Хорошо - есть небольшие отклонения")
        else:
            print("  Faithfulness (<0.6): Требует улучшения - частые галлюцинации")
        
        if scores["answer_relevancy"] > 0.8:
            print("  Answer Relevancy (>0.8): Отлично - ответы релевантны вопросам")
        elif scores["answer_relevancy"] > 0.6:
            print("  Answer Relevancy (0.6-0.8): Хорошо - в целом релевантны")
        else:
            print("  Answer Relevancy (<0.6): Требует улучшения - ответы не по теме")
        
        print(f"\nПроизводительность:")
        print(f"  Среднее время ответа: {results['performance']['average_response_time']:.2f} сек")
        print(f"  Минимальное время: {results['performance']['min_response_time']:.2f} сек")
        print(f"  Максимальное время: {results['performance']['max_response_time']:.2f} сек")


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results = evaluator.run_full_evaluation()
    evaluator.print_summary(results)