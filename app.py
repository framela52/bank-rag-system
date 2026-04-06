from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
import time
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent))

from llm.optimized_chain import OptimizedRAGChainWithStrategies
from config.settings import settings


# Lifespan для предзагрузки
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запуск
    print("Загрузка RAG системы с оптимизациями...")
    app.state.rag_chain = OptimizedRAGChainWithStrategies(use_compression=True, cache_ttl=3600)
    print("RAG система готова!")
    yield
    # Остановка
    print("Выключение RAG системы...")


app = FastAPI(
    title="Bank RAG System",
    description="Банковский консультант на основе RAG с оптимизациями",
    version="2.0.0",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    use_cache: Optional[bool] = True
    use_self_query: Optional[bool] = True      
    use_multi_query: Optional[bool] = True     
    use_rerank: Optional[bool] = True


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    context_used: bool
    num_docs: int
    response_time: Optional[float] = None
    from_cache: Optional[bool] = None


class BatchChatRequest(BaseModel):
    questions: List[str]
    k: Optional[int] = 5


class PerformanceStats(BaseModel):
    total_cached: int
    valid_entries: int
    expired_entries: int
    average_response_time: float
    total_requests: int


@app.get("/")
async def serve_frontend():
    """Отдаёт HTML интерфейс"""
    frontend_path = Path(__file__).parent / "templates" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Frontend not found"}


@app.get("/health")
async def health_check():
    """Проверка здоровья"""
    return {
        "status": "healthy",
        "model": settings.GIGACHAT_MODEL,
        "cache_stats": app.state.rag_chain.get_cache_stats()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Отправка вопроса с использованием кэша"""
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
        
        start_time = time.time()
        
        if request.use_cache:
            result = app.state.rag_chain.ask_with_cache(request.question, k=request.k)
        else:
            result = app.state.rag_chain.ask(request.question, k=request.k)
        
        response_time = time.time() - start_time
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            context_used=result["context_used"],
            num_docs=result["num_docs"],
            response_time=response_time,
            from_cache=result.get("from_cache", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/batch")
async def batch_chat(request: BatchChatRequest):
    """Пакетная обработка вопросов"""
    try:
        start_time = time.time()
        
        results = app.state.rag_chain.ask_batch(request.questions, k=request.k)
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "total_questions": len(request.questions),
            "total_time": total_time,
            "average_time": total_time / len(request.questions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/clear")
async def clear_history():
    """Очистка истории диалога и кэша"""
    try:
        app.state.rag_chain.clear_history()
        return {"message": "История диалога очищена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Очистка кэша ответов"""
    try:
        app.state.rag_chain.clear_cache()
        return {"message": "Кэш успешно очищен"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats", response_model=PerformanceStats)
async def get_cache_stats():
    """Получение статистики кэша"""
    try:
        stats = app.state.rag_chain.get_cache_stats()
        return PerformanceStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance")
async def get_performance():
    """Получение статистики производительности"""
    try:
        stats = app.state.rag_chain.get_cache_stats()
        return {
            "performance": stats,
            "cache_enabled": True,
            "model": settings.GIGACHAT_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("Запуск Bank RAG System API (оптимизированная версия)")
    print("="*60)
    print(f"API документация: http://localhost:8000/docs")
    print(f"Веб-интерфейс: http://localhost:8000")
    print("="*60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )