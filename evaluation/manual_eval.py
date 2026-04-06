import sys
from pathlib import Path
import json
import time
import os
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from llm.chain import RAGChain
from config.settings import settings


class ManualRAGEvaluator:
    """Оценка качества RAG системы с помощью ручных метрик"""
    
    def __init__(self):
        self.rag_chain = RAGChain(use_compression=True)
        self.test_questions = self._load_test_questions()
        self.stop_words = set(stopwords.words('russian') + stopwords.words('english'))
        
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
    
    def _preprocess_text(self, text: str) -> str:
        """Предобработка текста для анализа"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)
    
    def _calculate_claims(self, answer: str) -> List[str]:
        """Извлекает утверждения из ответа"""
        sentences = sent_tokenize(answer)
        claims = []
        for sent in sentences:
            processed = self._preprocess_text(sent)
            if len(processed.split()) > 3:  # Минимум 3 слова
                claims.append(processed)
        return claims
    
    def _tfidf_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """Вычисляет TF-IDF схожесть между двумя наборами текстов"""
        if not texts1 or not texts2:
            return np.array([[0.0]])
        
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        all_texts = texts1 + texts2
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        sim_matrix = cosine_similarity(tfidf_matrix[:len(texts1)], 
                                     tfidf_matrix[len(texts1):])
        return sim_matrix
    
    def calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Faithfulness: Доля утверждений в ответе, подкрепленных контекстом
        """
        claims = self._calculate_claims(answer)
        if not claims:
            return 1.0
        
        context_processed = [self._preprocess_text(ctx) for ctx in contexts]
        
        max_scores = []
        for claim in claims:
            sim_scores = self._tfidf_similarity([claim], context_processed)
            max_scores.append(np.max(sim_scores) if sim_scores.size > 0 else 0.0)
        
        return np.mean(max_scores)
    
    def calculate_answer_relevancy(self, question: str, answer: str, ground_truth: str) -> float:
        """
        Answer Relevancy: Релевантность ответа вопросу и эталонному ответу
        """
        q_processed = self._preprocess_text(question)
        a_processed = self._preprocess_text(answer)
        gt_processed = self._preprocess_text(ground_truth)
        
        # Комбинируем вопрос + эталонный ответ как референс
        reference = f"{q_processed} {gt_processed}"
        
        sim_scores = self._tfidf_similarity([a_processed], [reference])
        return float(sim_scores[0, 0])
    
    def calculate_context_relevancy(self, question: str, contexts: List[str]) -> float:
        """
        Context Relevancy: Доля релевантного контекста к вопросу
        """
        q_processed = self._preprocess_text(question)
        context_scores = []
        
        for context in contexts:
            c_processed = self._preprocess_text(context)
            sim_score = self._tfidf_similarity([q_processed], [c_processed])
            context_scores.append(float(sim_score[0, 0]))
        
        # Средневзвешенная релевантность (снижаем вес для менее релевантных чанков)
        total_score = sum(context_scores)
        return min(total_score / len(contexts), 1.0)
    
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
    
    def evaluate_manual(self, data: Dict) -> Dict:
        """Ручная оценка всех трех метрик"""
        print("\nРучная оценка метрик...")
        
        detailed_results = []
        faithfulness_scores = []
        answer_relevancy_scores = []
        context_relevancy_scores = []
        
        for i, (question, answer, contexts, ground_truth) in enumerate(
            zip(data["questions"], data["answers"], data["contexts"], data["ground_truths"])
        ):
            print(f"  Обработка {i+1}/{len(data['questions'])}: {question[:40]}...")
            
            faithfulness = self.calculate_faithfulness(answer, contexts)
            answer_relevancy = self.calculate_answer_relevancy(question, answer, ground_truth)
            context_relevancy = self.calculate_context_relevancy(question, contexts)
            
            faithfulness_scores.append(faithfulness)
            answer_relevancy_scores.append(answer_relevancy)
            context_relevancy_scores.append(context_relevancy)
            
            detailed_results.append({
                "question": question,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_relevancy": context_relevancy,
                "answer_preview": answer[:100] + "..."
            })
        
        return {
            "faithfulness": np.mean(faithfulness_scores),
            "answer_relevancy": np.mean(answer_relevancy_scores),
            "context_relevancy": np.mean(context_relevancy_scores),
            "detailed_results": detailed_results,
            "faithfulness_std": np.std(faithfulness_scores),
            "answer_relevancy_std": np.std(answer_relevancy_scores),
            "context_relevancy_std": np.std(context_relevancy_scores)
        }
    
    def run_full_evaluation(self) -> Dict:
        """Запускает полную оценку"""
        print("="*80)
        print("ОЦЕНКА КАЧЕСТВА RAG СИСТЕМЫ (РУЧНЫЕ МЕТРИКИ)")
        print("="*80)
        
        questions = [q["question"] for q in self.test_questions]
        
        data = self.collect_responses(questions)
        manual_scores = self.evaluate_manual(data)
        
        avg_time = sum(data["response_times"]) / len(data["response_times"])
        min_time = min(data["response_times"])
        max_time = max(data["response_times"])
        
        results = {
            "manual_scores": manual_scores,
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
        
        json_path = output_path / "manual_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nРезультаты сохранены в {json_path}")
    
    def print_summary(self, results: Dict):
        """Выводит сводку результатов"""
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ (РУЧНЫЕ МЕТРИКИ)")
        print("="*80)
        
        print("\nОсновные метрики:")
        print("-" * 40)
        scores = results["manual_scores"]
        
        print(f"  Faithfulness:      {scores['faithfulness']:.3f} ± {scores['faithfulness_std']:.3f}")
        print(f"  Answer Relevancy:  {scores['answer_relevancy']:.3f} ± {scores['answer_relevancy_std']:.3f}")
        print(f"  Context Relevancy: {scores['context_relevancy']:.3f} ± {scores['context_relevancy_std']:.3f}")
        
        print("\nИнтерпретация:")
        print("-" * 40)
        
        if scores["faithfulness"] > 0.8:
            print("  Faithfulness (>0.8): Отлично - ответы хорошо подкреплены контекстом")
        elif scores["faithfulness"] > 0.6:
            print("  Faithfulness (0.6-0.8): Хорошо - большинство утверждений подтверждены")
        else:
            print("  Faithfulness (<0.6): Нужна работа - много галлюцинаций")
        
        if scores["answer_relevancy"] > 0.8:
            print("  Answer Relevancy (>0.8): Отлично - ответы релевантны вопросу")
        elif scores["answer_relevancy"] > 0.6:
            print("  Answer Relevancy (0.6-0.8): Хорошо - в целом релевантны")
        else:
            print("  Answer Relevancy (<0.6): Плохо - ответы не по теме")
        
        if scores["context_relevancy"] > 0.8:
            print("  Context Relevancy (>0.8): Отлично - ретривер находит релевантный контекст")
        elif scores["context_relevancy"] > 0.6:
            print("  Context Relevancy (0.6-0.8): Хорошо - контекст в целом подходит")
        else:
            print("  Context Relevancy (<0.6): Плохо - ретривер возвращает нерелевантный контекст")
        
        print(f"\nПроизводительность:")
        print(f"  Среднее время ответа: {results['performance']['average_response_time']:.2f} сек")
        print(f"  Минимальное время: {results['performance']['min_response_time']:.2f} сек")
        print(f"  Максимальное время: {results['performance']['max_response_time']:.2f} сек")


