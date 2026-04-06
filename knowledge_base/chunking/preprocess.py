from pathlib import Path
from typing import List, Dict, Any
import re
import json

def clean_text(text: str) -> str:
    """Глубокая очистка банковских документов"""
    # Удаляем эмодзи
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols
        u"\U0001F680-\U0001F6FF"  # transport
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', ' ', text)
    
    # Убираем лишние markdown символы
    text = re.sub(r'[*]{2,}', '*', text)  
    text = re.sub(r'[_]{2,}', '_', text)  
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)      
    text = re.sub(r'\*(.*?)\*', r'\1', text)           
    text = re.sub(r'__(.*?)__', r'\1', text)           
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'[-]{3,}', '', text)                        
    text = re.sub(r'[=]{3,}', '', text)
    
    # Нормализуем markdown
    text = re.sub(r'\n{3,}', '\n\n', text)  
    text = re.sub(r'[ \t]+', ' ', text)     
    text = re.sub(r'\s+([.!?])', r'\1', text)   
    # Специальные символы БАНКА (оставляем важные)
    text = re.sub(r'[%€$₽№°]', ' ', text)                     
    text = re.sub(r'[—–-]', '-', text)
    return text.strip()

def load_and_clean_documents(directory: Path) -> List[Dict[str, Any]]:
    documents = []
    metadata_path = directory / "metadata.json"
    metadata = {}
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                metadata[item["source"]] = item
    
    for filepath in directory.glob("*.md"):
        if filepath.name == "metadata.json": continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_len = len(content)
        content = clean_text(content)
        cleaned_len = len(content)
        
        doc_metadata = metadata.get(filepath.name, {})
        documents.append({
            "title": doc_metadata.get("title", filepath.stem),
            "product_type": doc_metadata.get("product_type", "unknown"),
            "content": content,
            "source": filepath.name,
            "file_path": str(filepath),
            "chars_removed": original_len - cleaned_len
        })
    
    return documents