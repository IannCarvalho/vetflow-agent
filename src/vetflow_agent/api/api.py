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

# litellm é importado AQUI, no load do módulo (startup do processo), e NÃO
# dentro dos handlers: `import litellm` custa ~7s no primeiro load, e feito
# dentro de uma coroutine ele BLOQUEIA o event loop por esse tempo — com 1
# worker (uvicorn sem --workers no plano free do Render) o /health para de
# responder e a plataforma mata+reinicia o worker ("Shutting down" sem OOM nem
# deploy). Pagando o custo no boot, a porta só abre depois do import, então o
# health check nem começou. É dependência de runtime obrigatória
# (requirements.txt), então um ImportError aqui deve falhar o deploy (fail-fast),
# não virar erro silencioso por request.
import litellm  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402 - após load_dotenv
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from vetflow_agent.core.agent import run_agent  # noqa: E402
from vetflow_agent.core.provider import describe_provider, provider_available  # noqa: E402

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
    """Corpo do POST /chat.

    `model`/`api_key` são BYOK (bring your own key): a chave vive no navegador
    do usuário e chega por request. NUNCA é persistida nem logada pelo servidor;
    é usada só para a chamada ao LLM e descartada.
    """

    message: str = Field(..., min_length=1, description="Mensagem do usuário.")
    history: list[dict] = Field(
        default_factory=list,
        description="Histórico no formato OpenAI (role/content), opcional.",
    )
    model: str | None = Field(default=None, description="Modelo LiteLLM (BYOK), opcional.")
    api_key: str | None = Field(default=None, description="API key do usuário (BYOK), opcional.")


class ToolCall(BaseModel):
    name: str
    arguments: dict


class ChatResponse(BaseModel):
    ok: bool
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    error: str | None = None


# /health e /providers são `def` (não `async def`) DE PROPÓSITO: o FastAPI roda
# handlers síncronos no threadpool. Eles consultam a cadeia de credencial AWS
# (provider_available), que fora da AWS espera o timeout do IMDS (~2s) até o
# cache esquentar — dentro do event loop, isso travava toda request concorrente,
# incluindo o /chat, a cada health check do Render.
@app.get("/health")
def health() -> dict:
    """Status do serviço + se há provider utilizável."""
    return {"status": "ok", "provider_available": provider_available()}


@app.get("/providers")
def providers() -> dict:
    """Qual LLM será usado (rótulo legível, sem segredos)."""
    return {"available": provider_available(), "provider": describe_provider()}


class ModelsRequest(BaseModel):
    """Corpo do POST /providers/models (BYOK -- key usada só para listar)."""

    provider: str = Field(..., description="openai | anthropic | gemini")
    api_key: str = Field(..., min_length=1, description="API key do usuário (não persistida).")


# Famílias/padrões de modelos que NÃO são chat (não servem para o loop de tools):
# embeddings, geração de imagem/vídeo/áudio, TTS, moderação, AQA, etc.
_NON_CHAT_PATTERNS = (
    "embed",
    "embedding",
    "aqa",
    "image",
    "imagen",
    "vision-only",
    "tts",
    "audio",
    "whisper",
    "moderation",
    "veo",
    "dall-e",
    "text-embedding",
    "gemini-embedding",
    "learnlm",  # experimental, sem suporte consistente a tools
)


def _is_chat_model(model: str) -> bool:
    """True se o modelo aparenta suportar chat multiturn (heurística por nome)."""
    low = model.lower()
    return not any(p in low for p in _NON_CHAT_PATTERNS)


@app.post("/providers/models")
async def list_models(req: ModelsRequest) -> dict:
    """Lista os modelos REAIS que a key do usuário pode usar (custo zero).

    Usa litellm.get_valid_models(check_provider_endpoint=True), que chama a API
    de listagem do provider -- não gera tokens. A key vem no request (BYOK), é
    usada só para esta chamada e NÃO é persistida nem logada.
    """
    try:
        models = await run_in_threadpool(
            litellm.get_valid_models,
            check_provider_endpoint=True,
            custom_llm_provider=req.provider,
            api_key=req.api_key,
        )
        # Normaliza para o formato LiteLLM que o /chat espera (ex.: gemini/<id>).
        prefix = "gemini/" if req.provider == "gemini" else ""
        norm = []
        for m in models or []:
            name = m.split("/", 1)[-1] if "/" in m else m
            norm.append(f"{prefix}{name}" if prefix and not m.startswith(prefix) else m)
        # Mantém só modelos de CHAT (multiturn + tools). O endpoint de listagem
        # devolve também embeddings/imagem/tts/aqa, que quebram o /chat com
        # "Multiturn chat is not enabled" -- então filtramos por nome.
        norm = [m for m in norm if _is_chat_model(m)]
        # Dedup preservando ordem.
        seen: set[str] = set()
        unique = [x for x in norm if not (x in seen or seen.add(x))]
        if not unique:
            # get_valid_models engole erro de auth e devolve lista vazia; para o
            # usuário, lista vazia == key inválida ou sem acesso a modelos.
            return {
                "ok": False,
                "models": [],
                "error": "Chave inválida ou sem modelos disponíveis para este provedor.",
            }
        return {"ok": True, "models": unique, "error": None}
    except Exception as exc:  # noqa: BLE001 - erro de listagem vira resposta amigável
        return {"ok": False, "models": [], "error": f"{type(exc).__name__}: {exc}"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """Roda uma rodada do agente. Nunca levanta: erros vêm em ok=False/error.

    O JWT do usuário logado chega no header `Authorization: Bearer <jwt>` e é
    repassado ao agente/MCP (pass-through) — o chat age como esse usuário.
    """
    auth = request.headers.get("authorization", "")
    user_token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    result = await run_agent(
        req.message,
        history=req.history,
        model=req.model,
        api_key=req.api_key,
        user_token=user_token or None,
    )
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
