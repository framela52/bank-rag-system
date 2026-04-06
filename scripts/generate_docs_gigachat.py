import os
from pathlib import Path
import json
from typing import List, Dict
import requests
from dotenv import load_dotenv

load_dotenv()

class GigaChatClient:
    """Клиент для работы с GigaChat API"""
    
    def __init__(self):
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.client_id = os.getenv("GIGACHAT_CLIENT_ID")
        self.client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
        self.access_token = None
        
    def authenticate(self):
        """Аутентификация с отключением SSL проверки"""
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": os.getenv("RQ_UID", "123e4567-e89b-12d3-a456-426614174000")
        }
        
        data = {
            "scope": "GIGACHAT_API_PERS",
            "grant_type": "client_credentials"
        }
        
        auth = (self.client_id, self.client_secret)
        response = requests.post(
            self.auth_url, 
            headers=headers, 
            data=data, 
            auth=auth,
            verify=False  
        )
        
        if response.status_code == 200:
            self.access_token = response.json()["access_token"]
            print(" Аутентификация успешна!")
            return True
        else:
            print(f" Ошибка авторизации: {response.status_code} - {response.text}")
            return False
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Генерация текста через GigaChat"""
        if not self.access_token:
            self.authenticate()
        
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "RqUID": os.getenv("RQ_UID", "123e4567-e89b-12d3-a456-426614174000")
        }
        
        payload = {
            "model": "GigaChat",
            "messages": [
                {"role": "system", "content": "Ты - эксперт по банковским продуктам. Генерируй структурированную документацию в формате Markdown. Не используй эмодзи, пиши четко. Объем каждого документа не менее 1500 слов."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                verify=False,  
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f" GigaChat API ошибка: {e}")
            raise

def generate_bank_documents() -> List[Dict]:
    """Генерация документов о банковских продуктах через GigaChat"""
    
    client = GigaChatClient()
    documents = []
    
    # Zero-shot промпты для каждого типа документа
    prompts = [
        {
            "title": "Условия потребительского кредита",
            "product_type": "credit",
            "prompt": """
        Создай детальную документацию по потребительскому кредиту для банка. 
        Включи следующие разделы:
        1. Основные параметры кредита (сумма, срок, ставка)
        2. Требования к заемщику
        3. Необходимые документы
        4. Особые условия и преимущества

        Формат: Markdown. Используй списки и подзаголовки. Будь конкретен с цифрами.
        """
                },
                {
                    "title": "Депозит 'Надежный выбор'",
                    "product_type": "deposit",
                    "prompt": """
        Создай документацию по депозитному продукту для банка.
        Включи:
        1. Условия размещения (минимальная сумма, сроки)
        2. Процентные ставки в зависимости от срока
        3. Возможности пополнения и снятия
        4. Порядок начисления процентов

        Формат: Markdown. Укажи конкретные процентные ставки и суммы.
        """
                },
                {
                    "title": "Ипотечное кредитование",
                    "product_type": "mortgage",
                    "prompt": """
        Создай полную документацию по ипотечным программам банка.
        Включи:
        1. Программы ипотеки (новостройки, вторичка, семейная)
        2. Ставки и первоначальный взнос для каждой программы
        3. Требования к заемщикам и залоговому имуществу
        4. Документы для оформления

        Формат: Markdown. Укажи все параметры с конкретными значениями.
        """
                },
                {
                    "title": "Часто задаваемые вопросы",
                    "product_type": "faq",
                    "prompt": """
        Создай раздел FAQ для банковского консультанта.
        Включи 8-10 самых частых вопросов по:
        - Кредитам
        - Депозитам
        - Ипотеке
        - Картам
        - Обслуживанию

        Для каждого вопроса дай подробный ответ. Формат: Markdown.
        """
                },
                {
                    "title": "Тарифы и комиссии",
                    "product_type": "service",
                    "prompt": """
        Создай документацию по тарифам на банковское обслуживание.
        Включи:
        1. Тарифы РКО для разных пакетов
        2. Комиссии за переводы и снятие наличных
        3. Стоимость дополнительных услуг
        4. Льготные категории клиентов

        Формат: Markdown. Укажи конкретные суммы и проценты.
        """
                }
            ]
    
    for doc_config in prompts:
        print(f"Генерация документа: {doc_config['title']}")
        
        try:
            content = client.generate(doc_config["prompt"], temperature=0.3)
            
            documents.append({
                "title": doc_config["title"],
                "product_type": doc_config["product_type"],
                "content": content,
                "source": f"{doc_config['title'].lower().replace(' ', '_')}.md"
            })
            
        except Exception as e:
            print(f"Ошибка при генерации {doc_config['title']}: {e}")
            # Добавляем fallback контент
            documents.append(create_fallback_document(doc_config))
    
    return documents

def create_fallback_document(doc_config: Dict) -> Dict:
    """Создание fallback документа в случае ошибки API"""
    fallback_content = f"""
# {doc_config['title']}

## Общая информация
Документация по продукту временно недоступна. Пожалуйста, обратитесь в отделение банка.

**Тип продукта:** {doc_config['product_type']}

## Контактная информация
- Горячая линия: 8-800-XXX-XX-XX
- Часы работы: круглосуточно
- Отделения: ежедневно с 9:00 до 21:00
"""
    return {
        "title": doc_config["title"],
        "product_type": doc_config["product_type"],
        "content": fallback_content,
        "source": f"{doc_config['title'].lower().replace(' ', '_')}.md"
    }

def save_documents(documents: List[Dict], output_dir: Path):
    """Сохранение сгенерированных документов"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for doc in documents:
        filepath = output_dir / doc["source"]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doc["content"])
        
        print(f"Сохранен документ: {filepath}")
    
    # Сохраняем метаданные
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(
            [{"title": d["title"], "product_type": d["product_type"], "source": d["source"]} 
             for d in documents],
            f,
            ensure_ascii=False,
            indent=2
        )

if __name__ == "__main__":
    docs = generate_bank_documents()
    save_documents(docs, Path("data/raw"))