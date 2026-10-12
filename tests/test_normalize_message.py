"""Testes do seam de normalização da mensagem do LLM (agent.normalize_message).

Exercitam a fronteira direto, SEM montar o loop de run_agent: é exatamente a
superfície de teste que o refactor abriu. Cobrem os dois formatos que o LiteLLM
devolve (dict E objeto), arguments JSON válido e inválido, e as_openai_dict().
"""

from __future__ import annotations

from types import SimpleNamespace

from agent import AssistantMessage, ToolCall, normalize_message


# --------------------------------------------------------------------------- #
# Helpers para montar entradas cruas nos dois formatos.
# --------------------------------------------------------------------------- #
def _obj_message(content=None, tool_calls=None):
    """Mensagem como OBJETO do LiteLLM (com model_dump)."""
    m = SimpleNamespace(content=content, tool_calls=tool_calls)
    m.model_dump = lambda: {"role": "assistant", "content": content, "tool_calls": tool_calls}
    return m


def _obj_tool_call(call_id, name, arguments_json):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments_json),
    )


def _dict_tool_call(call_id, name, arguments_json):
    return {"id": call_id, "function": {"name": name, "arguments": arguments_json}}


# --------------------------------------------------------------------------- #
# Formato OBJETO
# --------------------------------------------------------------------------- #
def test_normalize_objeto_so_texto():
    msg = normalize_message(_obj_message(content="  Olá  "))
    assert isinstance(msg, AssistantMessage)
    assert msg.text == "Olá"  # strip aplicado
    assert msg.tool_calls == []


def test_normalize_objeto_com_tool_call_e_arguments_validos():
    raw = _obj_message(
        tool_calls=[_obj_tool_call("c1", "buscar_cliente", '{"nome": "Maria"}')]
    )
    msg = normalize_message(raw)
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "c1"
    assert tc.name == "buscar_cliente"
    assert tc.arguments == {"nome": "Maria"}


def test_normalize_objeto_text_vazio_quando_content_none():
    msg = normalize_message(_obj_message(content=None))
    assert msg.text == ""


# --------------------------------------------------------------------------- #
# Formato DICT
# --------------------------------------------------------------------------- #
def test_normalize_dict_com_tool_call():
    raw = {
        "content": "oi",
        "tool_calls": [_dict_tool_call("c9", "metricas_dashboard", "{}")],
    }
    msg = normalize_message(raw)
    assert msg.text == "oi"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "c9"
    assert msg.tool_calls[0].name == "metricas_dashboard"
    assert msg.tool_calls[0].arguments == {}


def test_normalize_dict_sem_tool_calls():
    msg = normalize_message({"content": "resposta final"})
    assert msg.text == "resposta final"
    assert msg.tool_calls == []


# --------------------------------------------------------------------------- #
# arguments inválido -> {} (mesmo tratamento tolerante de antes)
# --------------------------------------------------------------------------- #
def test_normalize_arguments_json_invalido_vira_dict_vazio():
    raw = _obj_message(tool_calls=[_obj_tool_call("c1", "t", "{nao e json valido")])
    msg = normalize_message(raw)
    assert msg.tool_calls[0].arguments == {}


def test_normalize_arguments_none_vira_dict_vazio():
    raw = _obj_message(tool_calls=[_obj_tool_call("c1", "t", None)])
    msg = normalize_message(raw)
    assert msg.tool_calls[0].arguments == {}


def test_normalize_arguments_ja_dict_e_preservado():
    """Alguns providers podem mandar arguments já como dict (não string)."""
    raw = {"tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": {"a": 1}}}]}
    msg = normalize_message(raw)
    assert msg.tool_calls[0].arguments == {"a": 1}


# --------------------------------------------------------------------------- #
# as_openai_dict() reproduz o model_dump()/dict(message) original
# --------------------------------------------------------------------------- #
def test_as_openai_dict_usa_model_dump_do_objeto():
    raw = _obj_message(content="x", tool_calls=[_obj_tool_call("c1", "t", "{}")])
    anexo = normalize_message(raw).as_openai_dict()
    # as_openai_dict() devolve o resultado de model_dump() do objeto cru.
    assert anexo["role"] == "assistant"
    assert anexo["content"] == "x"
    assert anexo["tool_calls"] == raw.model_dump()["tool_calls"]


def test_as_openai_dict_para_entrada_dict_e_o_proprio_dict():
    raw = {"role": "assistant", "content": "y", "tool_calls": None}
    msg = normalize_message(raw)
    assert msg.as_openai_dict() == raw


def test_as_openai_dict_e_anexavel_ao_historico():
    """O dict devolvido deve ter o papel de assistant para o histórico OpenAI."""
    msg = normalize_message(_obj_message(content="z"))
    anexo = msg.as_openai_dict()
    assert anexo["role"] == "assistant"
    assert anexo["content"] == "z"
