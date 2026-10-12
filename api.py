"""API HTTP do vetflow-agent (FastAPI).

Expõe o loop do agente (`agent.run_agent`) como endpoints para o frontend do
VetFlow consumir. O Streamlit continua existindo como UI alternativa; esta API é
o caminho "produto": o React do VetFlow chama `POST /chat` e renderiza o chat
com a identidade visual dele.

Endpoints:
- GET  /health     -> status do serviço + provider ativo.
- GET  /providers  -> qual LLM será usado (sem revelar segredos).
- POST /chat        -> {message, history?} -> {ok, text, tool_calls, error}.

Modelo de concorrência: cada /chat abre sua própria sessão MCP (subprocesso
stdio) via run_agent e a fecha ao terminar. Simples e seguro para uso de
portfólio; um pool de sessões persistentes é uma otimização futura, necessária
só sob carga concorrente real.

Config por ambiente:
- CORS_ORIGINS: lista separada por vírgula de origens permitidas
  (default "http://localhost:3039,http://localhost:5173" -- dev do VetFlow/Vite).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Carrega o .env ANTES de importar o agente, para que ANTHROPIC/OPENAI/GEMINI/
# AWS e VETFLOW_* cheguem também ao subprocesso MCP.
load_dotenv()

from fastapi import FastAPI  # noqa: E402 - após load_dotenv
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agent import run_agent  # noqa: E402
from provider import describe_provider, provider_available  # noqa: E402

app = FastAPI(title="VetFlow Agent API", version="1.0.0")

_default_origins = "http://localhost:3039,http://localhost:5173"
_origins = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Corpo do POST /chat."""

    message: str = Field(..., min_length=1, description="Mensagem do usuário.")
    history: list[dict] = Field(
        default_factory=list,
        description="Histórico no formato OpenAI (role/content), opcional.",
    )


class ToolCall(BaseModel):
    name: str
    arguments: dict


class ChatResponse(BaseModel):
    ok: bool
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    error: str | None = None


@app.get("/health")
async def health() -> dict:
    """Status do serviço + se há provider utilizável."""
    return {"status": "ok", "provider_available": provider_available()}


@app.get("/providers")
async def providers() -> dict:
    """Qual LLM será usado (rótulo legível, sem segredos)."""
    return {"available": provider_available(), "provider": describe_provider()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Roda uma rodada do agente. Nunca levanta: erros vêm em ok=False/error."""
    result = await run_agent(req.message, history=req.history)
    tool_calls = [
        ToolCall(name=c["name"], arguments=c.get("arguments") or {})
        for c in result.tool_calls
    ]
    return ChatResponse(
        ok=result.ok,
        text=result.text,
        tool_calls=tool_calls,
        error=result.error,
    )
