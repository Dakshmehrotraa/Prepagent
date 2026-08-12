from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.orchestrator import run_agent_turn
from app.retrieval.retriever import get_retriever
from app.models.classifier import get_classifier

router = APIRouter()


class AgentRequest(BaseModel):
    message: str


class ClassifyRequest(BaseModel):
    text: str


class RetrieveRequest(BaseModel):
    query: str
    k: int = 3


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/agent")
def agent_endpoint(req: AgentRequest):
    if not req.message.strip():
        raise HTTPException(400, "message cannot be empty")
    result = run_agent_turn(req.message)
    return result


@router.post("/classify")
def classify_endpoint(req: ClassifyRequest):
    classifier = get_classifier()
    return classifier.predict(req.text)


@router.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    retriever = get_retriever()
    return {"results": retriever.search(req.query, k=req.k)}
