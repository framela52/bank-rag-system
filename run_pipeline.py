import sys
from pathlib import Path
import subprocess
import os

def print_step(step_num, step_name):
    print("\n" + "="*60)
    print(f"STEP {step_num}: {step_name}")
    print("="*60)

def run_script(script_path):
    return subprocess.run([sys.executable, script_path]).returncode == 0

def main():
    print("="*60)
    print("RUN RAG SYSTEM PIPELINE")
    print("="*60)
    
    # STEP 0: Установка зависимостей
    print_step("0", "Install dependencies")
    if os.path.exists("requirements.txt"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed")
    else:
        print("requirements.txt not found, skipping")
    
    # Создание директорий
    for d in ["data/processed", "data/vector_stores", "evaluation/results", "data/cache"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # Этап 1: Подготовка данных
    print_step("1.2", "Chunking (сравнение стратегий)")
    from evaluation.chunking_eval import run_comprehensive_evaluation
    run_comprehensive_evaluation()
    
    print_step("1.3", "Embeddings + FAISS")
    run_script("build_knowledge_base.py")
    
    # Этап 2: Система ретрива
    print_step("2.3", "Оценка ретрива (Hit Rate, MRR)")
    run_script("evaluation/retrieval_metrics.py")
    
    # Этап 3: LLM интеграция
    print_step("3.2", "RAG цепочка с GigaChat")
    run_script("llm/chain.py")
    
    print_step("3.3", "Оптимизации (self-query, multi-query, rerank)")
    run_script("llm/optimization.py")
    
    # Этап 4: Анализ
    print_step("4.1", "Оценка faithfulness, answer_relevancy (ручная)")
    run_script("evaluation/run_full_evaluation.py")
    
    print_step("4.2", "Тест производительности с кэшем")
    run_script("llm/optimized_chain.py")
    
    print("\n" + "="*60)
    print("\nДля проверки работы системы:")
    print("  1. Запустите сервер: python app.py")
    print("  2. Откройте: http://localhost:8000")
    print("\nДокументация API: http://localhost:8000/docs")

if __name__ == "__main__":
    main()