"""docs-seeker - /v1/chat 路由"""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from docs_seeker.api.deps import get_chat_service
from docs_seeker.api.schemas import ChatRequest, ChatResponse, SourceDoc

router = APIRouter(tags=["chat"])


def _sse(event: dict) -> str:
    """将事件字典序列化为 SSE 帧（data: <json>，空行结尾）"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _history(req: ChatRequest) -> list[dict] | None:
    return (
        [{"role": m.role, "content": m.content} for m in req.conversation_history] if req.conversation_history else None
    )


@router.post("/chat")
def chat(req: ChatRequest):
    """问答接口。

    - stream=True（默认）：SSE 流式输出（text/event-stream），事件依次为
      meta（检索来源）→ delta*（增量文本）→ done（完整回答与元数据）；异常时输出 error 事件。
    - stream=False：一次性 JSON 返回 ChatResponse（兼容旧客户端与评估服务）。

    同步 handler + 同步生成器：FastAPI/Starlette 自动放入线程池执行，
    检索/Milvus/Redis/LLM 等阻塞调用不会阻塞事件循环。
    """
    service = get_chat_service()
    history = _history(req)

    if not req.stream:
        result = service.chat(
            req.question,
            history=history,
            top_k=req.top_k,
            use_cache=req.use_cache,
            session_id=req.session_id,
            user_id=req.user_id,
        )
        return ChatResponse(
            answer=result.answer,
            confidence=result.confidence,
            sources=[SourceDoc(**s) for s in result.sources],
            cached=result.cached,
            query_decomposed=result.query_decomposed,
        )

    return StreamingResponse(
        (
            _sse(event)
            for event in service.chat_stream(
                req.question,
                history=history,
                top_k=req.top_k,
                use_cache=req.use_cache,
                session_id=req.session_id,
                user_id=req.user_id,
            )
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 禁用代理/nginx 缓冲，避免 SSE 帧被聚合后失去实时性
            "X-Accel-Buffering": "no",
        },
    )
