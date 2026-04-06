import pytest
import json
from pathlib import Path
from knowledge_base.chunking.strategies import ChunkingStrategies
from knowledge_base.chunking.preprocess import load_and_clean_documents

@pytest.fixture
def sample_documents():
    """Пример документов для тестов"""
    return [
        {
            "title": "Тест кредит",
            "product_type": "credit",
            "content": """
# Условия кредита
Сумма: 100 000 - 5 000 000 руб.
Ставка: 12.5% годовых.
Срок: 1-5 лет.

## Документы
- Паспорт
- Справка 2-НДФЛ
- СНИЛС
            """,
            "source": "test_credit.md"
        }
    ]

def test_fixed_size_chunking(sample_documents):
    chunker = ChunkingStrategies(chunk_size=100, chunk_overlap=10)
    chunks = chunker.by_fixed_size(sample_documents)
    
    assert len(chunks) > 0
    assert all(len(c["text"]) <= 100 for c in chunks)
    assert "fixed_size" in chunks[0]["metadata"]["chunk_strategy"]

def test_recursive_chunking(sample_documents):
    chunker = ChunkingStrategies(chunk_size=150, chunk_overlap=20)
    chunks = chunker.recursive_split(sample_documents)
    
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["chunk_strategy"] == "recursive"

def test_metadata_preservation(sample_documents):
    chunker = ChunkingStrategies()
    chunks = chunker.by_sentences(sample_documents)
    
    assert chunks[0]["metadata"]["title"] == "Тест кредит"
    assert chunks[0]["metadata"]["product_type"] == "credit"
