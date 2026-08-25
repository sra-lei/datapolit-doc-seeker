"""docs-seeker - /v1/retrieve 路由"""

from fastapi import APIRouter

from docs_seeker.api.deps import get_search_service
from docs_seeker.api.schemas import RetrieveRequest, RetrieveResponse, SourceDoc

router = APIRouter(tags=["retrieve"])


# 同步 handler：FastAPI 自动放入线程池执行，避免阻塞事件循环
@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    docs = get_search_service().search(req.query, top_k=req.top_k, use_summary=req.use_summary)
    return RetrieveResponse(
        docs=[SourceDoc(**{k: v for k, v in d.items() if k in SourceDoc.model_fields}) for d in docs],
        total=len(docs),
    )
