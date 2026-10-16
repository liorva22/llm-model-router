from datetime import UTC, datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .budget import make_ledger, month_key
from .config import ROOT, load_config
from .core import Router
from .difficulty import DifficultyClassifier, load_dataset
from .llm import make_llm


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)


def create_app(cfg=None, llm=None, ledger=None) -> FastAPI:
    cfg = cfg or load_config()
    # Trains in well under a second on the bundled 200 labeled queries, so there's no model artifact to ship.
    clf = DifficultyClassifier().fit(*load_dataset(ROOT / "data" / "queries.jsonl"))
    router = Router(cfg, llm or make_llm(cfg["mock"]), ledger or make_ledger(), clf.predict_proba)
    app = FastAPI(title="LLM Model Router", version="1.0.0")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "mock": cfg["mock"]}

    @app.post("/v1/chat")
    def chat(req: ChatRequest):
        return router.handle(req.query)

    @app.get("/v1/budget")
    def budget():
        now = datetime.now(UTC)
        spent = router.ledger.month_spend(now)
        limit = cfg["budget"]["monthly_usd"]
        return {"month": month_key(now), "spent_usd": spent, "limit_usd": limit, "fraction": spent / limit}

    return app
