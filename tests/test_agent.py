"""Testes de agent.py (loop via LiteLLM, formato OpenAI).

Determinísticos: litellm.acompletion e a sessão MCP são mockados. Nenhuma rede,
nenhum subprocesso, nenhuma chave real.
"""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from types import SimpleNamespace

import agent
from agent import AgentResult, run_agent


def _install_litellm_stub(monkeypatch, responses):
    """Instala um módulo litellm falso com acompletion() programável."""
    mod = types.ModuleType("litellm")
    state = {"i": 0, "calls": []}

    async def acompletion(**kwargs):
        state["calls"].append(kwargs)
        resp = responses[state["i"]]
        state["i"] += 1
        return resp

    mod.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return state


def _msg(content=None, tool_calls=None):
    """Monta um objeto message no formato que o agent lê (choices[0].message)."""
    m = SimpleNamespace(content=content, tool_calls=tool_calls)
    m.model_dump = lambda: {"role": "assistant", "content": content, "tool_calls": tool_calls}
    return SimpleNamespace(choices=[SimpleNamespace(message=m)])


def _tool_call(call_id, name, arguments_json):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments_json),
    )


# --------------------------------------------------------------------------- #
# AgentResult
# --------------------------------------------------------------------------- #
def test_agent_result_ok_default_true():
    assert AgentResult(text="x").ok is True


def test_agent_result_tool_calls_independente_por_instancia():
    a = AgentResult(text="a")
    b = AgentResult(text="b")
    a.tool_calls.append({"name": "t"})
    assert a.tool_calls == [{"name": "t"}]
    assert b.tool_calls == []


# --------------------------------------------------------------------------- #
# run_agent sem nenhum provider disponível
# --------------------------------------------------------------------------- #
async def test_run_agent_sem_provider_retorna_ok_false_sem_levantar(monkeypatch):
    from provider import ProviderUnavailable

    # Garante um litellm importável (não deve nem chegar a usá-lo).
    _install_litellm_stub(monkeypatch, [])

    def _boom(model=None, api_key=None):
        raise ProviderUnavailable("Nenhum provider de LLM configurado.")

    monkeypatch.setattr(agent, "select_model_with_overrides", _boom)

    result = await run_agent("oi")

    assert isinstance(result, AgentResult)
    assert result.ok is False
    assert "provider de LLM" in (result.error or "")


# --------------------------------------------------------------------------- #
# Loop de tool-use (formato OpenAI/LiteLLM) mockado
# --------------------------------------------------------------------------- #
async def test_run_agent_loop_tool_use_mockado(monkeypatch):
    # 1a resposta: pede a tool metricas_dashboard; 2a: texto final.
    responses = [
        _msg(tool_calls=[_tool_call("c1", "metricas_dashboard", "{}")]),
        _msg(content="Resposta final"),
    ]
    _install_litellm_stub(monkeypatch, responses)

    def _sel(model=None, api_key=None):
        return SimpleNamespace(model="gpt-4o", provider="openai")

    monkeypatch.setattr(agent, "select_model_with_overrides", _sel)

    fake_session = SimpleNamespace()

    @asynccontextmanager
    async def fake_connect_session(user_token=None):
        yield fake_session

    monkeypatch.setattr(agent, "connect_session", fake_connect_session)
    monkeypatch.setattr(agent, "list_tools", lambda s: _async_return([]))
    monkeypatch.setattr(agent, "tools_to_openai", lambda tools: [])

    async def fake_dispatch_tool(session, name, arguments):
        assert session is fake_session
        return ("ok", False)

    monkeypatch.setattr(agent, "dispatch_tool", fake_dispatch_tool)

    eventos: list[tuple[str, dict]] = []
    result = await run_agent(
        "quantos clientes?", on_tool_event=lambda n, a: eventos.append((n, a))
    )

    assert result.ok is True
    assert result.text == "Resposta final"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "metricas_dashboard"
    assert eventos == [("metricas_dashboard", {})]


async def test_run_agent_parsa_argumentos_json(monkeypatch):
    """Argumentos vêm como string JSON e devem ser parseados para dict."""
    responses = [
        _msg(tool_calls=[_tool_call("c1", "buscar_cliente", '{"nome": "Maria"}')]),
        _msg(content="pronto"),
    ]
    _install_litellm_stub(monkeypatch, responses)
    def _sel(model=None, api_key=None):
        return SimpleNamespace(model="gpt-4o", provider="openai")

    monkeypatch.setattr(agent, "select_model_with_overrides", _sel)

    fake_session = SimpleNamespace()

    @asynccontextmanager
    async def fake_connect_session(user_token=None):
        yield fake_session

    monkeypatch.setattr(agent, "connect_session", fake_connect_session)
    monkeypatch.setattr(agent, "list_tools", lambda s: _async_return([]))
    monkeypatch.setattr(agent, "tools_to_openai", lambda tools: [])

    captured = {}

    async def fake_dispatch_tool(session, name, arguments):
        captured["args"] = arguments
        return ("ok", False)

    monkeypatch.setattr(agent, "dispatch_tool", fake_dispatch_tool)

    result = await run_agent("acha a Maria")
    assert result.ok is True
    assert captured["args"] == {"nome": "Maria"}


async def _async_return(value):
    return value
