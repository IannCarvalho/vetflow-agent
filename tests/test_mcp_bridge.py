"""Testes unitários de mcp_bridge.py.

Determinísticos: sem backend real, sem rede, sem subir o subprocesso MCP. As
funções puras são testadas direto; dispatch_tool recebe uma fake session com
call_tool assíncrono.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# ExceptionGroup é builtin a partir do 3.11. Em 3.10 usamos o backport
# `exceptiongroup` (dependência de teste), para o CI rodar em todas as versões.
try:  # pragma: no cover - caminho depende da versão do Python
    ExceptionGroup  # type: ignore[used-before-def]  # noqa: B018, F821
except NameError:  # pragma: no cover
    from exceptiongroup import ExceptionGroup  # type: ignore[no-redef]

from mcp_bridge import (
    MCPConnectionError,
    _check_required_env,
    _flatten_tool_result,
    _friendly_detail,
    _unwrap_exception,
    dispatch_tool,
    tool_to_openai,
)


# --------------------------------------------------------------------------- #
# _unwrap_exception
# --------------------------------------------------------------------------- #
def test_unwrap_exception_desce_grupo_aninhado_ate_folha():
    leaf = ValueError("causa real")
    inner = ExceptionGroup("inner", [leaf])
    outer = ExceptionGroup("unhandled errors in a TaskGroup", [inner])

    assert _unwrap_exception(outer) is leaf


def test_unwrap_exception_folha_simples_retorna_a_si():
    leaf = RuntimeError("x")
    assert _unwrap_exception(leaf) is leaf


# --------------------------------------------------------------------------- #
# _friendly_detail
# --------------------------------------------------------------------------- #
def test_friendly_detail_usa_ultima_linha_nao_vazia_do_stderr():
    stderr = "linha 1\nlinha 2 real\n   \n\n"
    detail = _friendly_detail(ValueError("ignorado"), stderr)
    assert detail == "linha 2 real"


def test_friendly_detail_stderr_vazio_cai_para_str_da_folha():
    leaf = ValueError("mensagem da folha")
    assert _friendly_detail(leaf, "") == "mensagem da folha"


def test_friendly_detail_stderr_vazio_e_folha_sem_str_usa_classe():
    class Vazia(Exception):
        def __str__(self) -> str:  # str() vazio força o fallback de classe
            return ""

    assert _friendly_detail(Vazia(), "   ") == "Vazia"


# --------------------------------------------------------------------------- #
# _check_required_env
# --------------------------------------------------------------------------- #
def test_check_required_env_sem_as_tres_levanta_com_detail_listando_as_tres(monkeypatch):
    for var in ("VETFLOW_BASE_URL", "VETFLOW_EMAIL", "VETFLOW_PASSWORD"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(MCPConnectionError) as exc_info:
        _check_required_env()

    detail = exc_info.value.detail or ""
    assert "VETFLOW_BASE_URL" in detail
    assert "VETFLOW_EMAIL" in detail
    assert "VETFLOW_PASSWORD" in detail


def test_check_required_env_com_as_tres_nao_levanta(monkeypatch):
    monkeypatch.setenv("VETFLOW_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("VETFLOW_EMAIL", "a@b.com")
    monkeypatch.setenv("VETFLOW_PASSWORD", "segredo")

    # Não deve levantar.
    assert _check_required_env() is None


# --------------------------------------------------------------------------- #
# _flatten_tool_result
# --------------------------------------------------------------------------- #
def test_flatten_tool_result_concatena_textos_com_newline():
    result = SimpleNamespace(
        content=[SimpleNamespace(text="a"), SimpleNamespace(text="b")],
        structured_content=None,
    )
    assert _flatten_tool_result(result) == "a\nb"


def test_flatten_tool_result_content_vazio_usa_structured_content():
    result = SimpleNamespace(content=[], structured_content={"x": 1})
    assert _flatten_tool_result(result) == str({"x": 1})


def test_flatten_tool_result_tudo_vazio_retorna_string_vazia():
    result = SimpleNamespace(content=[], structured_content=None)
    assert _flatten_tool_result(result) == ""


# --------------------------------------------------------------------------- #
# dispatch_tool (async)
# --------------------------------------------------------------------------- #
class _FakeSession:
    """Fake de ClientSession com call_tool async programável."""

    def __init__(self, *, result=None, raise_exc=None):
        self._result = result
        self._raise_exc = raise_exc
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._result


async def test_dispatch_tool_sucesso_retorna_texto_e_false():
    result = SimpleNamespace(
        is_error=False,
        content=[SimpleNamespace(text="resposta ok")],
        structured_content=None,
    )
    session = _FakeSession(result=result)

    text, is_error = await dispatch_tool(session, "metricas_dashboard", {})

    assert text == "resposta ok"
    assert is_error is False
    assert session.calls == [("metricas_dashboard", {})]


async def test_dispatch_tool_is_error_true_propaga_flag():
    result = SimpleNamespace(
        is_error=True,
        content=[SimpleNamespace(text="deu ruim")],
        structured_content=None,
    )
    session = _FakeSession(result=result)

    text, is_error = await dispatch_tool(session, "t", {"a": 1})

    assert is_error is True
    assert text == "deu ruim"


async def test_dispatch_tool_excecao_vira_erro_amigavel_sem_propagar():
    session = _FakeSession(raise_exc=RuntimeError("boom"))

    text, is_error = await dispatch_tool(session, "buscar_cliente", {})

    assert is_error is True
    assert "buscar_cliente" in text  # nome da tool aparece na mensagem


# --------------------------------------------------------------------------- #
# tool_to_openai (formato function-calling usado pelo LiteLLM)
# --------------------------------------------------------------------------- #
def test_tool_to_openai_formato_function():
    tool = SimpleNamespace(
        name="buscar_cliente",
        description="  Busca clientes.  ",
        input_schema={"type": "object", "properties": {"nome": {"type": "string"}}},
    )
    out = tool_to_openai(tool)
    assert out["type"] == "function"
    assert out["function"]["name"] == "buscar_cliente"
    assert out["function"]["description"] == "Busca clientes."
    assert out["function"]["parameters"] == tool.input_schema


def test_tool_to_openai_fallback_schema_quando_none():
    tool = SimpleNamespace(name="x", description="d", input_schema=None)
    out = tool_to_openai(tool)
    assert out["function"]["parameters"] == {"type": "object", "properties": {}}


# --------------------------------------------------------------------------- #
# Modo in-process (VETFLOW_MCP_INPROCESS) — consumo sem subprocesso.
# --------------------------------------------------------------------------- #
from mcp_bridge import _inprocess_enabled, _InProcessSession  # noqa: E402


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
     ("", False), ("0", False), ("false", False), ("off", False)],
)
def test_inprocess_enabled_le_o_flag(monkeypatch, value, expected):
    monkeypatch.setenv("VETFLOW_MCP_INPROCESS", value)
    assert _inprocess_enabled() is expected


def test_inprocess_enabled_ausente_e_false(monkeypatch):
    monkeypatch.delenv("VETFLOW_MCP_INPROCESS", raising=False)
    assert _inprocess_enabled() is False


class _FakeInProcessServer:
    """Fake de MCPServer: list_tools/call_tool async, como o MCPServer real."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return [
            SimpleNamespace(name="buscar_cliente", description="d", input_schema={}),
            SimpleNamespace(name="metricas_dashboard", description="d", input_schema={}),
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            is_error=False,
            content=[SimpleNamespace(text=f"ok:{name}")],
            structured_content=None,
        )


async def test_inprocess_session_list_tools_expoe_atributo_tools():
    """O adaptador normaliza list_tools para um objeto com .tools (como o protocolo)."""
    session = _InProcessSession(_FakeInProcessServer())
    result = await session.list_tools()
    assert [t.name for t in result.tools] == ["buscar_cliente", "metricas_dashboard"]


async def test_inprocess_session_funciona_com_os_helpers_do_bridge():
    """A sessão in-process casa com list_tools/dispatch_tool/tools_to_openai do bridge."""
    from mcp_bridge import dispatch_tool as _dispatch
    from mcp_bridge import list_tools as _list
    from mcp_bridge import tools_to_openai as _to_openai

    server = _FakeInProcessServer()
    session = _InProcessSession(server)

    tools = await _list(session)
    openai_tools = _to_openai(tools)
    assert openai_tools[0]["type"] == "function"
    assert {t["function"]["name"] for t in openai_tools} == {
        "buscar_cliente",
        "metricas_dashboard",
    }

    text, is_error = await _dispatch(session, "buscar_cliente", {"nome": "ana"})
    assert text == "ok:buscar_cliente"
    assert is_error is False
    assert server.calls == [("buscar_cliente", {"nome": "ana"})]
