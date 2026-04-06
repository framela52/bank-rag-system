import requests

API_URL = "http://localhost:8000"

def ask_question(question: str):
    """Отправляет вопрос к RAG системе."""
    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question, "k": 5}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Вопрос: {question}")
        print(f"Ответ: {result['answer']}")
        print(f"Источники: {[s['title'] for s in result['sources']]}")
        print(f"Использовано документов: {result['num_docs']}")
        print("-" * 50)
    else:
        print(f"Ошибка: {response.status_code} - {response.text}")

def ask_optimized(question: str):
    """Отправляет вопрос с оптимизациями (self-query + multi-query)."""
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "question": question,
            "k": 5,
            "use_cache": True,
            "use_self_query": True,
            "use_multi_query": True,
            "use_rerank": False   # отключаем реранкинг (по результатам тестов)
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Вопрос: {question}")
        print(f"Ответ: {result['answer']}")
        print(f"Источники: {[s['title'] for s in result['sources']]}")
        print(f"Время ответа: {result.get('response_time', 'N/A')} сек")
        print(f"Из кэша: {result.get('from_cache', False)}")
        print("-" * 50)
    else:
        print(f"Ошибка: {response.status_code} - {response.text}")

def clear_history():
    """Очищает историю диалога."""
    response = requests.post(f"{API_URL}/chat/clear")
    print(response.json()["message"])


def clear_cache():
    """Очищает кэш ответов."""
    response = requests.post(f"{API_URL}/cache/clear")
    print(response.json()["message"])


def get_cache_stats():
    """Получает статистику кэша."""
    response = requests.get(f"{API_URL}/cache/stats")
    if response.status_code == 200:
        stats = response.json()
        print("Статистика кэша:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print(f"Ошибка: {response.status_code}")


if __name__ == "__main__":
    print("="*60)
    print("ТЕСТ RAG СИСТЕМЫ")
    print("="*60)
    
    # Обычные вопросы
    print("\n1. ОБЫЧНЫЕ ВОПРОСЫ:")
    questions = [
        "Какая минимальная сумма для открытия депозита?",
        "Можно ли досрочно погасить кредит без штрафов?",
        "Какой первоначальный взнос по ипотеке на новостройки?"
    ]
    
    for q in questions:
        ask_question(q)
    
    # Оптимизированные вопросы (self-query + multi-query)
    print("\n2. ОПТИМИЗИРОВАННЫЕ ВОПРОСЫ (self-query + multi-query):")
    for q in questions:
        ask_optimized(q)
    
    # Статистика кэша
    print("\n3. СТАТИСТИКА КЭША:")
    get_cache_stats()
    
    # чищаем историю (не кэш)
    print("\n4. ОЧИСТКА ИСТОРИИ:")
    clear_history()