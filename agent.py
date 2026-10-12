"""Loop do agente Claude sobre as ferramentas do `vetflow-mcp`.

Implementa o protocolo de tool-use da Anthropic: pede uma resposta ao modelo,
executa cada `tool_use` via a ponte MCP, devolve os `tool_result` com o
`tool_use_id` casando cada resultado ao pedido, e repete até o modelo parar de
pedir ferramentas. Toda falha vira um `AgentResult(ok=False, ...)` amigável --
nenhum traceback cru escapa para a UI.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mcp_bridge import (
    MCPConnectionError,
    connect_session,
    dispatch_tool,
    list_tools,
    tools_to_anthropic,
)

# Trava de segurança: no máximo 12 idas e voltas de tool-use por mensagem, para
# o modelo não entrar num laço infinito de chamadas de ferramenta.
MAX_TURNS = 12

MODEL = "claude-3-5-sonnet-latest"
MAX_TOKENS = 2048

SYSTEM_PROMPT = (
    "Você é o assistente do VetFlow, um sistema de gestão de clínica "
    "veterinária. Responda sempre em português do Brasil, de forma objetiva. "
    "Use as ferramentas disponíveis para consultar dados reais da clínica "
    "ANTES de responder -- nunca invente clientes, pets, datas ou números. "
    "Se uma ferramenta retornar erro, explique o erro ao usuário em linguagem "
    "simples em vez de inventar um resultado."
)


@dataclass
class AgentResult:
    """Resultado de uma rodada do agente para uma mensagem do usuário."""

    text: str
    ok: bool = True
    error: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def _extract_text(content_blocks: list[Any]) -> str:
    """Concatena os blocos de texto da resposta final do modelo."""
    parts: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(p for p in parts if p).strip()


async def run_agent(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    on_tool_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> AgentResult:
    """Roda uma rodada do agente para `user_message`.

    Args:
        user_message: a mensagem do usuário.
        history: histórico de mensagens no formato da Anthropic (role/content).
        on_tool_event: callback chamado ANTES de cada tool ser despachada,
            recebendo (nome, argumentos) -- usado pela UI para mostrar a chamada.

    Returns:
        AgentResult com o texto final e a lista de tool_calls, ou ok=False com
        uma mensagem de erro amigável (nunca levanta).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return AgentResult(
            text="",
            ok=False,
            error=(
                "ANTHROPIC_API_KEY não está definida. Configure a chave no .env "
                "(ou no ambiente) para conversar com o assistente."
            ),
        )

    try:
        import anthropic
    except ImportError:
        return AgentResult(
            text="",
            ok=False,
            error="Pacote 'anthropic' não instalado. Rode: pip install -r requirements.txt",
        )

    tool_calls: list[dict[str, Any]] = []

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)

        async with connect_session() as session:
            mcp_tools = await list_tools(session)
            anthropic_tools = tools_to_anthropic(mcp_tools)

            messages: list[dict[str, Any]] = list(history or [])
            messages.append({"role": "user", "content": user_message})

            for _ in range(MAX_TURNS):
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=anthropic_tools,
                )

                if response.stop_reason != "tool_use":
                    return AgentResult(
                        text=_extract_text(response.content),
                        ok=True,
                        tool_calls=tool_calls,
                    )

                # O modelo pediu ferramentas: anexa a resposta dele e despacha
                # cada bloco tool_use, devolvendo os resultados casados por id.
                messages.append({"role": "assistant", "content": response.content})

                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    name = block.name
                    arguments = block.input or {}
                    if on_tool_event is not None:
                        on_tool_event(name, arguments)
                    tool_calls.append({"name": name, "arguments": arguments})

                    text, is_error = await dispatch_tool(session, name, arguments)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": text,
                            "is_error": is_error,
                        }
                    )

                messages.append({"role": "user", "content": tool_results})

            # Esgotou MAX_TURNS sem o modelo concluir.
            return AgentResult(
                text="",
                ok=False,
                error=(
                    f"O agente excedeu o limite de {MAX_TURNS} rodadas de "
                    "ferramentas sem concluir. Tente reformular a pergunta."
                ),
                tool_calls=tool_calls,
            )

    except MCPConnectionError as exc:
        return AgentResult(text="", ok=False, error=str(exc), tool_calls=tool_calls)
    except anthropic.AnthropicError as exc:
        return AgentResult(
            text="",
            ok=False,
            error=f"Erro na API da Anthropic: {exc}",
            tool_calls=tool_calls,
        )
    except Exception as exc:  # noqa: BLE001 - última barreira: nada cru vaza
        return AgentResult(
            text="",
            ok=False,
            error=f"Erro inesperado no agente: {exc}",
            tool_calls=tool_calls,
        )
