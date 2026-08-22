"""docs-seeker - HTTP 路由"""
from fastapi import APIRouter
from loguru import logger
from docs_seeker.api.schemas import ChatRequest, ChatResponse, RetrieveRequest, RetrieveResponse, HealthResponse, SourceDoc
from docs_seeker.api.deps import get_composite_retriever, get_generator, get_query_decomposer
from docs_seeker.infra.guard import check_injection, sanitize_output
from docs_seeker.infra.semantic_cache import get_semantic_cache
from docs_seeker.infra.milvus_store import get_milvus_store

router = APIRouter(prefix="/v1", tags=["docs-seeker"])


@router.get("/health", response_model=HealthResponse)
async def health():
    milvus_ok, redis_ok = False, False
    try:
        store = get_milvus_store()
        milvus_ok = store.has_collection(store.collection_name)
    except Exception:
        pass
    try:
        redis_ok = get_semantic_cache()._available
    except Exception:
        pass
    return HealthResponse(status="ok", milvus_connected=milvus_ok, redis_connected=redis_ok)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    docs = get_composite_retriever().search(req.query, top_k=req.top_k, use_summary=req.use_summary)
    return RetrieveResponse(docs=[SourceDoc(**{k: v for k, v in d.items() if k in SourceDoc.model_fields}) for d in docs], total=len(docs))


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    ok, reason = check_injection(req.question)
    if not ok:
        return ChatResponse(answer=reason, confidence="low", sources=[])

    if req.use_cache:
        cached = get_semantic_cache().search(req.question)
        if cached:
            return ChatResponse(answer=cached["answer"], confidence=cached.get("confidence","medium"), sources=[SourceDoc(**s) for s in cached.get("sources",[])], cached=True)

    sub_questions = get_query_decomposer().decompose(req.question)
    retriever = get_composite_retriever()
    all_docs = []
    for sq in sub_questions:
        all_docs.extend(retriever.search(sq, top_k=req.top_k))

    seen, deduped = set(), []
    for d in all_docs:
        did = d.get("id", "")
        if did not in seen:
            seen.add(did)
            deduped.append(d)
    deduped = deduped[:req.top_k]

    history = [{"role": m.role, "content": m.content} for m in req.conversation_history] if req.conversation_history else None
    answer, confidence = get_generator().generate(req.question, deduped, history)
    answer = sanitize_output(answer)

    if req.use_cache:
        fields = ("id","text","source","chapter","chapter_title","section","section_title","score","sources")
        get_semantic_cache().store(req.question, {"answer": answer, "confidence": confidence, "sources": [{k: v for k, v in d.items() if k in fields} for d in deduped]})

    fields = ("id","text","source","chapter","chapter_title","section","section_title","score","sources")
    sources = [SourceDoc(**{k: v for k, v in d.items() if k in fields}) for d in deduped]
    return ChatResponse(answer=answer, confidence=confidence, sources=sources, cached=False, query_decomposed=sub_questions if len(sub_questions) > 1 else None)
