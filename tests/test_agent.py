"""Testes de agent.py.

Determinísticos: Anthropic e a sessão MCP são mockados. Nenhuma rede, nenhum
subprocesso, nenhuma chave real. O loop de tool-use é exercitado com um
AsyncAnthropic fake que devolve primeiro um stop_reason="tool_use" e depois um
"end_turn".
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import agent
from agent import AgentResult, _extract_text, run_agent


# --------------------------------------------------------------------------- #
# run_agent sem chave
# --------------------------------------------------------------------------- #
async def test_run_agent_sem_api_key_retorna_ok_false_sem_levantar(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = await run_agent("oi")

    assert isinstance(result, AgentResult)
    assert result.ok is False
    assert "ANTHROPIC_API_KEY" in (result.error or "")


# --------------------------------------------------------------------------- #
# _extract_text
# --------------------------------------------------------------------------- #
def test_extract_text_pega_so_blocos_de_texto():
    blocks = [
        SimpleNamespace(type="text", text="oi"),
        SimpleNamespace(type="tool_use"),
    ]
    assert _extract_text(blocks) == "oi"


def test_extract_text_junta_varios_textos_com_newline():
    blocks = [
        SimpleNamespace(type="text", text="linha 1"),
        SimpleNamespace(type="text", text="linha 2"),
    ]
    assert _extract_text(blocks) == "linha 1\nlinha 2"


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
    assert b.tool_calls == []  # sem aliasing do default mutável


# --------------------------------------------------------------------------- #
# Loop tool-use com Anthropic mockado
# --------------------------------------------------------------------------- #
class _FakeMessages:
    """Fake de client.messages com create() async programável por rodadas."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        resp = self._responses[self._i]
        self._i += 1
        return resp


class _FakeAsyncAnthropic:
    """Fake de anthropic.AsyncAnthropic que devolve respostas pré-programadas."""

    def __init__(self, *args, **kwargs):
        # 1a rodada: pede tool_use; 2a rodada: encerra.
        self.messages = _FakeMessages(
            [
                SimpleNamespace(
                    stop_reason="tool_use",
                    content=[
                        SimpleNamespace(
                            type="tool_use",
                            name="metricas_dashboard",
                            input={},
                            id="t1",
                        )
                    ],
                ),
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text="Resposta final")],
                ),
            ]
        )


async def test_run_agent_loop_tool_use_mockado(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-fake")

    # Anthropic: substitui AsyncAnthropic no módulo que agent.py importa por
    # dentro (seja o real instalado ou o stub do conftest).
    anthropic_mod = sys.modules["anthropic"]
    monkeypatch.setattr(anthropic_mod, "AsyncAnthropic", _FakeAsyncAnthropic)

    # Sessão MCP: connect_session vira um CM assíncrono que cede uma fake session.
    fake_session = SimpleNamespace()

    @asynccontextmanager
    async def fake_connect_session():
        yield fake_session

    monkeypatch.setattr(agent, "connect_session", fake_connect_session)

    async def fake_list_tools(session):
        assert session is fake_session
        return []

    monkeypatch.setattr(agent, "list_tools", fake_list_tools)
    monkeypatch.setattr(agent, "tools_to_anthropic", lambda tools: [])

    async def fake_dispatch_tool(session, name, arguments):
        assert session is fake_session
        return ("ok", False)

    monkeypatch.setattr(agent, "dispatch_tool", fake_dispatch_tool)

    eventos: list[tuple[str, dict]] = []

    def on_tool_event(name, arguments):
        eventos.append((name, arguments))

    result = await run_agent("quantos clientes?", on_tool_event=on_tool_event)

    assert result.ok is True
    assert result.text == "Resposta final"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "metricas_dashboard"
    assert len(eventos) == 1
    assert eventos[0][0] == "metricas_dashboard"
