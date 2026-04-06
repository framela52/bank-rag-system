from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter
)
from typing import List, Dict, Any
from pathlib import Path
import hashlib


class ChunkingStrategies:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def by_fixed_size(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        return self._split_documents(documents, text_splitter, "fixed_size")
    
    def by_sentences(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? ", ";", ", ", " "],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        return self._split_documents(documents, text_splitter, "by_sentences")
    
    def recursive_split(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n# ", "\n## ", "\n### ", "\n", ". ", " "],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        return self._split_documents(documents, text_splitter, "recursive")
    
    def by_markdown(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return self._split_documents(documents, text_splitter, "markdown")
    
    def _split_documents(self, documents: List[Dict], splitter, strategy_name: str) -> List[Dict]:
        all_chunks = []
        for doc in documents:
            split_texts = splitter.split_text(doc["content"])
            for i, text in enumerate(split_texts):
                if not text.strip():
                    continue
                chunk_id = hashlib.md5(f"{doc['source']}_{i}_{text[:100]}".encode()).hexdigest()[:12]
                chunk = {
                    "id": chunk_id,
                    "text": text.strip(),
                    "metadata": {
                        "source": doc["source"],
                        "title": doc["title"],
                        "product_type": doc["product_type"],
                        "chunk_index": i,
                        "chunk_strategy": strategy_name,
                        "total_chunks": len(split_texts)
                    }
                }
                all_chunks.append(chunk)
        return all_chunks

def compare_chunking_strategies(documents: List[Dict], strategies: List[str] = None):
    """Базовое сравнение стратегий"""
    if strategies is None:
        strategies = ["fixed_size", "by_sentences", "recursive", "markdown"]
    
    chunker = ChunkingStrategies()
    results = {}
    
    strategy_methods = {
        "fixed_size": chunker.by_fixed_size,
        "by_sentences": chunker.by_sentences,
        "recursive": chunker.recursive_split,
        "markdown": chunker.by_markdown
    }
    
    for strategy in strategies:
        if strategy in strategy_methods:
            chunks = strategy_methods[strategy](documents)
            results[strategy] = {
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0,
                "chunks": chunks
            }
    
    print("\n=== БАЗОВЫЕ МЕТРИКИ ЧАНКИНГА ===")
    for strategy, data in results.items():
        print(f"{strategy}: {data['num_chunks']} чанков, {data['avg_chunk_size']:.0f} симв.")
    
    return results

def save_chunks(chunks: List[Dict], output_dir: Path, strategy_name: str):
    from pathlib import Path
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_file = output_dir / f"chunks_{strategy_name}.json"
    import json
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(chunks)} чанков для стратегии {strategy_name}")
    return chunks_file
