"""Testes da API FastAPI (api.py).

Determinísticos: run_agent é mockado, então não há subprocesso MCP, rede nem
chave real. Usa o TestClient do FastAPI (síncrono).
"""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - fastapi é dependência de runtime
    TestClient = None

pytestmark = pytest.mark.skipif(TestClient is None, reason="fastapi não instalado")


def _client(monkeypatch, agent_result):
    """Monta um TestClient com run_agent/provider mockados."""
    import vetflow_agent.api.api as api_mod
    import vetflow_agent.core.agent as agent_mod

    async def fake_run_agent(
        message, history=None, on_tool_event=None, model=None, api_key=None, user_token=None
    ):
        return agent_result

    monkeypatch.setattr(api_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(api_mod, "provider_available", lambda: True)
    monkeypatch.setattr(api_mod, "describe_provider", lambda: "openai (gpt-4o)")
    return TestClient(api_mod.app), agent_mod


def test_health_ok(monkeypatch):
    from vetflow_agent.core.agent import AgentResult

    client, _ = _client(monkeypatch, AgentResult(text="x"))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["provider_available"] is True


def test_providers_descreve_sem_segredos(monkeypatch):
    from vetflow_agent.core.agent import AgentResult

    client, _ = _client(monkeypatch, AgentResult(text="x"))
    r = client.get("/providers")
    body = r.json()
    assert body["available"] is True
    assert body["provider"] == "openai (gpt-4o)"


def test_chat_ok_retorna_texto_e_tools(monkeypatch):
    from vetflow_agent.core.agent import AgentResult

    result = AgentResult(
        text="3 clientes e 5 pets.",
        ok=True,
        tool_calls=[{"name": "metricas_dashboard", "arguments": {}}],
    )
    client, _ = _client(monkeypatch, result)
    r = client.post("/chat", json={"message": "quantos clientes?"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["text"] == "3 clientes e 5 pets."
    assert body["tool_calls"] == [{"name": "metricas_dashboard", "arguments": {}}]
    assert body["error"] is None


def test_chat_erro_vem_em_ok_false(monkeypatch):
    from vetflow_agent.core.agent import AgentResult

    result = AgentResult(text="", ok=False, error="Nenhum provider de LLM configurado.")
    client, _ = _client(monkeypatch, result)
    r = client.post("/chat", json={"message": "oi"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "provider" in body["error"]


def test_chat_mensagem_vazia_e_422(monkeypatch):
    from vetflow_agent.core.agent import AgentResult

    client, _ = _client(monkeypatch, AgentResult(text="x"))
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 422  # validação do Pydantic (min_length=1)


def test_is_chat_model_filtra_nao_chat():
    import vetflow_agent.api.api as api_mod

    assert api_mod._is_chat_model("gemini/gemini-2.0-flash") is True
    assert api_mod._is_chat_model("gemini/gemini-embedding-001") is False
    assert api_mod._is_chat_model("text-embedding-3-large") is False
    assert api_mod._is_chat_model("gpt-4o") is True


def test_providers_models_lista_filtrada(monkeypatch):
    import sys
    import types

    from fastapi.testclient import TestClient

    import vetflow_agent.api.api as api_mod

    litellm_stub = types.ModuleType("litellm")
    litellm_stub.get_valid_models = lambda **kw: [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-embedding-001",  # deve ser filtrado
    ]
    # O api.py importa o litellm no topo: o stub vale no atributo do módulo, não
    # em sys.modules (que só afetaria um `import litellm` feito depois).
    monkeypatch.setattr(api_mod, "litellm", litellm_stub)
    monkeypatch.setitem(sys.modules, "litellm", litellm_stub)

    client = TestClient(api_mod.app)
    r = client.post("/providers/models", json={"provider": "gemini", "api_key": "x"})
    body = r.json()
    assert body["ok"] is True
    assert body["models"] == ["gemini/gemini-2.0-flash"]  # embedding removido


def test_providers_models_lista_vazia_e_invalida(monkeypatch):
    import sys
    import types

    from fastapi.testclient import TestClient

    import vetflow_agent.api.api as api_mod

    litellm_stub = types.ModuleType("litellm")
    litellm_stub.get_valid_models = lambda **kw: []  # key inválida -> litellm devolve vazio
    monkeypatch.setattr(api_mod, "litellm", litellm_stub)
    monkeypatch.setitem(sys.modules, "litellm", litellm_stub)

    client = TestClient(api_mod.app)
    r = client.post("/providers/models", json={"provider": "gemini", "api_key": "bogus"})
    body = r.json()
    assert body["ok"] is False
    assert body["models"] == []
    assert body["error"]


def test_chat_repassa_bearer_como_user_token(monkeypatch):
    import vetflow_agent.api.api as api_mod
    from vetflow_agent.core.agent import AgentResult

    captured = {}

    async def fake_run_agent(
        message, history=None, on_tool_event=None, model=None, api_key=None, user_token=None
    ):
        captured["user_token"] = user_token
        return AgentResult(text="ok")

    monkeypatch.setattr(api_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(api_mod, "provider_available", lambda: True)
    monkeypatch.setattr(api_mod, "describe_provider", lambda: "x")

    from fastapi.testclient import TestClient

    client = TestClient(api_mod.app)
    r = client.post(
        "/chat",
        json={"message": "oi"},
        headers={"Authorization": "Bearer jwt-user-123"},
    )
    assert r.status_code == 200
    assert captured["user_token"] == "jwt-user-123"


# --------------------------------------------------------------------------- #
# _normalize_model_ids (pura: sem TestClient nem litellm)
# --------------------------------------------------------------------------- #
def test_normalize_model_ids_gemini_prefixa_filtra_e_deduplica():
    import vetflow_agent.api.api as api_mod

    crus = [
        "models/gemini-3.8-flash",
        "gemini/gemini-3.8-flash",  # duplicata depois de prefixar
        "models/gemini-embedding-001",  # não é chat
        "models/gemini-3.6-flash",
    ]
    assert api_mod._normalize_model_ids(crus, "gemini") == [
        "gemini/gemini-3.8-flash",
        "gemini/gemini-3.6-flash",
    ]


def test_normalize_model_ids_outros_providers_mantem_o_id():
    import vetflow_agent.api.api as api_mod

    crus = ["gpt-6-luna", "text-embedding-3-large", "o4-mini", "gpt-6-luna", "tts-1"]
    assert api_mod._normalize_model_ids(crus, "openai") == ["gpt-6-luna", "o4-mini"]


def test_normalize_model_ids_lista_ausente_vira_vazia():
    import vetflow_agent.api.api as api_mod

    assert api_mod._normalize_model_ids(None, "anthropic") == []
