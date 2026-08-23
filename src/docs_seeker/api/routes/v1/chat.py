"""docs-seeker - /v1/chat 路由"""
from fastapi import APIRouter

from docs_seeker.api.deps import get_chat_service
from docs_seeker.api.schemas import ChatRequest, ChatResponse, SourceDoc

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.conversation_history] if req.conversation_history else None
    result = get_chat_service().chat(req.question, history=history, top_k=req.top_k, use_cache=req.use_cache)
    return ChatResponse(
        answer=result.answer,
        confidence=result.confidence,
        sources=[SourceDoc(**s) for s in result.sources],
        cached=result.cached,
        query_decomposed=result.query_decomposed,
    )
