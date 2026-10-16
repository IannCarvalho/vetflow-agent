"""Contrato com o vetflow-mcp REAL (o pacote de vendor/vetflow-mcp, não um fake).

Os outros testes da ponte usam sessões falsas. Estes conferem que as tools do
servidor de verdade chegam ao formato function-calling com o schema dos
argumentos. Se o SDK `mcp` renomear o atributo do schema (no 1.x era
`inputSchema`), `tool_to_openai` cai no schema vazio em silêncio e o modelo
passa a chamar as tools sem argumento nenhum.

Pulados quando o vetflow_mcp não está instalado no venv; o CI o instala a partir
do submodule.
"""

from __future__ import annotations

import pytest

from vetflow_agent.core.mcp_bridge import _InProcessSession, list_tools, tools_to_openai

mcp_server = pytest.importorskip("vetflow_mcp.server")


async def _functions() -> dict[str, dict]:
    tools = await list_tools(_InProcessSession(mcp_server.mcp))
    return {f["function"]["name"]: f["function"] for f in tools_to_openai(tools)}


async def test_tools_reais_chegam_com_o_schema_dos_argumentos():
    funcs = await _functions()
    busca = funcs["buscar_cliente"]["parameters"]
    assert "nome" in busca["properties"]
    assert busca["required"] == ["nome"]
    assert "pet_id" in funcs["ficha_do_pet"]["parameters"]["properties"]
    assert "data" in funcs["agenda_do_dia"]["parameters"]["properties"]
    assert all(f["description"] for f in funcs.values())


async def test_inprocess_expoe_so_as_tools_de_leitura():
    # As de escrita só são registradas pelo main() do MCP (modo subprocesso, com
    # VETFLOW_ENABLE_WRITES=true). O in-process, que é o modo do deploy, nunca
    # as expõe: se isso mudar, que seja de propósito.
    funcs = await _functions()
    assert "buscar_cliente" in funcs
    escrita = {"criar_cliente", "atualizar_cliente", "criar_pet", "registrar_consulta"}
    assert not escrita & funcs.keys()
