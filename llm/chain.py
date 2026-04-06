from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
import requests
import uuid
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from knowledge_base.retriever.compression import HybridRetrieverWithCompression


class GigaChatClient:
    """Клиент для работы с GigaChat API"""
    
    def __init__(self):
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.client_id = settings.GIGACHAT_CLIENT_ID
        self.client_secret = settings.GIGACHAT_CLIENT_SECRET
        self.access_token = None
        
        # Отключаем SSL предупреждения
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def authenticate(self) -> bool:
        """Аутентификация с отключением SSL проверки"""
        print("Аутентификация в GigaChat...")
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4())
        }
        
        data = {
            "scope": "GIGACHAT_API_PERS",
            "grant_type": "client_credentials"
        }
        
        auth = (self.client_id, self.client_secret)
        
        try:
            response = requests.post(
                self.auth_url, 
                headers=headers, 
                data=data, 
                auth=auth,
                verify=False,
                timeout=30.0
            )
            
            if response.status_code == 200:
                self.access_token = response.json()["access_token"]
                print("Аутентификация успешна")
                return True
            else:
                print(f"Ошибка авторизации: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Ошибка при аутентификации: {e}")
            return False
    
    def generate(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 1000) -> str:
        """Генерация текста через GigaChat"""
        if not self.access_token:
            if not self.authenticate():
                raise Exception("Не удалось получить токен GigaChat")
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "RqUID": str(uuid.uuid4())
        }
        
        payload = {
            "model": settings.GIGACHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                verify=False,
                timeout=60.0
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API ошибка: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Ошибка при запросе к GigaChat: {e}")


def format_docs(docs: List) -> str:
    """Форматирует документы для контекста."""
    if not docs:
        return "Информация не найдена."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'неизвестный документ')
        title = doc.metadata.get('title', source)
        
        formatted.append(f"[Источник {i}: {title}]\n{doc.page_content}\n")
    
    return "\n---\n".join(formatted)


def extract_sources(docs: List) -> List[Dict]:
    """Извлекает источники для цитирования."""
    sources = []
    seen = set()
    
    for doc in docs:
        source = doc.metadata.get('source', 'неизвестный документ')
        title = doc.metadata.get('title', source)
        
        if title not in seen:
            seen.add(title)
            sources.append({
                "title": title,
                "product_type": doc.metadata.get('product_type', ''),
                "source": source
            })
    
    return sources


class RAGChain:
    """RAG цепочка для ответа на вопросы через GigaChat API."""
    
    def __init__(self, use_compression: bool = True):
        print("Инициализация RAG цепочки с GigaChat...")
        
        self.client = GigaChatClient()
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        
        self.retriever = HybridRetrieverWithCompression(
            use_compression=use_compression,
            compression_k=settings.RETRIEVAL_K
        )
        self.chat_history = []
        print("RAG цепочка готова!")
    
    def _get_system_prompt(self, context: str, chat_history: str) -> str:
        """Формирует системный промпт без эмодзи."""
        return f"""Ты - профессиональный банковский консультант. Отвечай на вопросы клиента, используя ТОЛЬКО информацию из документов ниже.

Документы:
{context}

История диалога:
{chat_history}

Правила:
1. Отвечай только на основе предоставленных документов.
2. Не используй эмодзи и смайлики в ответах.
3. Если информация отсутствует в документах, честно скажи: "Извините, я не могу найти эту информацию в документации банка. Пожалуйста, обратитесь в поддержку."
4. Приводи конкретные цифры, даты и условия из документов.
5. Обязательно указывай источник информации (название документа).
6. Будь вежливым, понятным и профессиональным.
7. Не давай советы, которые могут навредить клиенту.
8. Отвечай на русском языке.
9. Используй только факты из предоставленных документов, не добавляй информацию из своего знания.

Вопрос клиента: """

    def _format_chat_history(self) -> str:
        """Форматирует историю диалога для системного промпта."""
        if not self.chat_history:
            return "Нет истории диалога."
        
        formatted = []
        for msg in self.chat_history[-6:]:
            if msg["role"] == "user":
                formatted.append(f"Клиент: {msg['content']}")
            elif msg["role"] == "assistant":
                formatted.append(f"Консультант: {msg['content']}")
        
        return "\n".join(formatted)
    
    def ask(self, question: str, k: int = None) -> Dict[str, Any]:
        """Задаёт вопрос и возвращает ответ с источниками."""
        
        if k is None:
            k = settings.RETRIEVAL_K
        
        print(f"\nПоиск документов по запросу: {question[:50]}...")
        
        # 1. Поиск релевантных документов
        docs = self.retriever.hybrid_search(question, k=k)
        
        if not docs:
            return {
                "answer": "Извините, я не могу найти информацию по вашему вопросу в документации банка. Пожалуйста, обратитесь в поддержку.",
                "sources": [],
                "context_used": False
            }
        
        print(f"Найдено {len(docs)} документов")
        
        # Форматируем контекст
        context = format_docs(docs)
        
        # Форматируем историю
        chat_history_str = self._format_chat_history()
        
        # Формируем системный промпт
        system_prompt = self._get_system_prompt(context, chat_history_str)
        
        # Формируем сообщения для API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        print("Генерация ответа через GigaChat API...")
        
        # Запрос к GigaChat API
        try:
            answer = self.client.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Сохраняем в историю
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # Извлекаем источники
            sources = extract_sources(docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": True,
                "num_docs": len(docs)
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"Ошибка: {error_msg}")
            
            return {
                "answer": f"Произошла ошибка при обращении к GigaChat API: {error_msg}",
                "sources": [],
                "context_used": False,
                "error": error_msg
            }
    
    def clear_history(self):
        """Очищает историю диалога."""
        self.chat_history = []
        print("История диалога очищена")
    
    def ask_with_sources(self, question: str, k: int = None) -> str:
        """Возвращает ответ с явными ссылками на источники."""
        result = self.ask(question, k)
        
        if not result["context_used"]:
            return result["answer"]
        
        answer = result["answer"]
        if result["sources"]:
            answer += "\n\nИсточники:"
            for source in result["sources"]:
                answer += f"\n- {source['title']}"
        
        return answer


# Тестовый скрипт
if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    print("="*80)
    print("ТЕСТ RAG ЦЕПОЧКИ (GigaChat API)")
    print("="*80)
    
    # Проверка настроек
    print(f"\nClient ID: {settings.GIGACHAT_CLIENT_ID[:15] if settings.GIGACHAT_CLIENT_ID else 'НЕ ЗАДАН'}...")
    print(f"Client Secret: {'*' * 15 if settings.GIGACHAT_CLIENT_SECRET else 'НЕ ЗАДАН'}")
    print(f"Модель: {settings.GIGACHAT_MODEL}")
    print(f"Temperature: {settings.LLM_TEMPERATURE}")
    
    if not settings.GIGACHAT_CLIENT_ID or not settings.GIGACHAT_CLIENT_SECRET:
        print("\nОШИБКА: GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET не заданы в .env файле")
    else:
        try:
            # Инициализация
            rag = RAGChain(use_compression=True)
            
            # Тестовые вопросы
            test_questions = [
                "Какая минимальная сумма для открытия депозита?",
                "Можно ли досрочно погасить кредит без штрафов?",
                "Какой первоначальный взнос по ипотеке на новостройки?"
            ]
            
            for i, question in enumerate(test_questions, 1):
                print(f"\n{'='*60}")
                print(f"Вопрос {i}: {question}")
                print('='*60)
                
                try:
                    result = rag.ask(question)
                    print(f"\nОтвет:\n{result['answer']}")
                    if result.get('sources'):
                        print(f"\nИсточники:")
                        for source in result['sources']:
                            print(f"   - {source['title']}")
                except Exception as e:
                    print(f"Ошибка: {e}")
                
                print("-"*60)
                
        except Exception as e:
            print(f"\nКритическая ошибка: {e}")