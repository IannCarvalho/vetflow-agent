"""Testes unitários de mcp_bridge.py.

Determinísticos: sem backend real, sem rede, sem subir o subprocesso MCP. As
funções puras são testadas direto; dispatch_tool recebe uma fake session com
call_tool assíncrono.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mcp_bridge
from mcp_bridge import (
    MCPConnectionError,
    _check_required_env,
    _flatten_tool_result,
    _friendly_detail,
    _unwrap_exception,
    dispatch_tool,
    tool_to_anthropic,
    tools_to_anthropic,
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
# tool_to_anthropic / tools_to_anthropic
# --------------------------------------------------------------------------- #
def test_tool_to_anthropic_chaves_exatas_e_description_strip():
    tool = SimpleNamespace(
        name="metricas_dashboard",
        description="  conta coisas  ",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )

    result = tool_to_anthropic(tool)

    assert set(result.keys()) == {"name", "description", "input_schema"}
    assert result["name"] == "metricas_dashboard"
    assert result["description"] == "conta coisas"
    assert result["input_schema"] == {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }


def test_tool_to_anthropic_input_schema_none_usa_fallback():
    tool = SimpleNamespace(name="t", description="d", input_schema=None)
    result = tool_to_anthropic(tool)
    assert result["input_schema"] == {"type": "object", "properties": {}}


def test_tool_to_anthropic_description_none_vira_string_vazia():
    tool = SimpleNamespace(name="t", description=None, input_schema={})
    result = tool_to_anthropic(tool)
    assert result["description"] == ""


def test_tools_to_anthropic_converte_lista():
    tools = [
        SimpleNamespace(name="a", description="da", input_schema={"type": "object"}),
        SimpleNamespace(name="b", description=None, input_schema=None),
    ]

    result = tools_to_anthropic(tools)

    assert [t["name"] for t in result] == ["a", "b"]
    assert result[0]["input_schema"] == {"type": "object"}
    assert result[1]["input_schema"] == {"type": "object", "properties": {}}


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
