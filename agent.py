"""Loop do agente sobre as ferramentas do `vetflow-mcp`, via LiteLLM.

O LiteLLM dá UMA interface (formato OpenAI) sobre Claude, GPT, Gemini e Bedrock,
então o loop de tool-use é escrito uma vez só, em formato OpenAI:
- o modelo responde com `message.tool_calls`;
- cada tool é despachada via a ponte MCP;
- devolvemos uma mensagem `{"role": "tool", "tool_call_id": ..., "content": ...}`
  por chamada, casando o id;
- paramos quando não há mais `tool_calls`.

Qual provider/modelo é usado vem de `provider.select_model()` (cascata de
credenciais). Toda falha vira `AgentResult(ok=False, ...)` amigável -- nenhum
traceback cru escapa para a UI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mcp_bridge import (
    MCPConnectionError,
    connect_session,
    dispatch_tool,
    list_tools,
    tools_to_openai,
)
from provider import ProviderUnavailable, select_model_with_overrides

# Trava de segurança: no máximo 12 idas e voltas de tool-use por mensagem.
MAX_TURNS = 12
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


def _message_text(message: Any) -> str:
    """Extrai o texto da mensagem do assistente (dict ou objeto do LiteLLM)."""
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return (content or "").strip()


def _get_tool_calls(message: Any) -> list[Any]:
    """Lista de tool_calls da mensagem (vazia se não houver)."""
    if isinstance(message, dict):
        return message.get("tool_calls") or []
    return getattr(message, "tool_calls", None) or []


def _tc_field(tc: Any, *path: str) -> Any:
    """Acessa um campo de um tool_call, tolerante a dict vs objeto."""
    cur = tc
    for key in path:
        cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
        if cur is None:
            return None
    return cur


async def run_agent(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    on_tool_event: Callable[[str, dict[str, Any]], None] | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> AgentResult:
    """Roda uma rodada do agente para `user_message` (nunca levanta).

    `model`/`api_key` são o caminho BYOK: quando vêm do request (chave do
    navegador do usuário), são usados só nesta chamada e NUNCA persistidos.
    """
    try:
        import litellm
    except ImportError:
        return AgentResult(
            text="",
            ok=False,
            error="Pacote 'litellm' não instalado. Rode: pip install -r requirements.txt",
        )

    tool_calls: list[dict[str, Any]] = []

    try:
        selection = select_model_with_overrides(model=model, api_key=api_key)
        chosen_model = selection.model
        # Chave BYOK (se veio no request) é passada ao litellm só nesta chamada.
        byok_key = (api_key or "").strip() or None

        async with connect_session() as session:
            mcp_tools = await list_tools(session)
            openai_tools = tools_to_openai(mcp_tools)

            messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history or [])
            messages.append({"role": "user", "content": user_message})

            for _ in range(MAX_TURNS):
                completion_kwargs: dict[str, Any] = {
                    "model": chosen_model,
                    "messages": messages,
                    "tools": openai_tools,
                    "max_tokens": MAX_TOKENS,
                }
                if byok_key:
                    completion_kwargs["api_key"] = byok_key
                response = await litellm.acompletion(**completion_kwargs)
                message = response.choices[0].message
                requested = _get_tool_calls(message)

                if not requested:
                    return AgentResult(
                        text=_message_text(message),
                        ok=True,
                        tool_calls=tool_calls,
                    )

                # Anexa a mensagem do assistente (com os tool_calls) e despacha.
                messages.append(
                    message.model_dump() if hasattr(message, "model_dump") else dict(message)
                )

                for tc in requested:
                    call_id = _tc_field(tc, "id")
                    name = _tc_field(tc, "function", "name")
                    raw_args = _tc_field(tc, "function", "arguments") or "{}"
                    try:
                        if isinstance(raw_args, str):
                            arguments = json.loads(raw_args)
                        else:
                            arguments = dict(raw_args)
                    except (ValueError, TypeError):
                        arguments = {}

                    if on_tool_event is not None:
                        on_tool_event(name, arguments)
                    tool_calls.append({"name": name, "arguments": arguments})

                    text, _is_error = await dispatch_tool(session, name, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": text,
                        }
                    )

            return AgentResult(
                text="",
                ok=False,
                error=(
                    f"O agente excedeu o limite de {MAX_TURNS} rodadas de "
                    "ferramentas sem concluir. Tente reformular a pergunta."
                ),
                tool_calls=tool_calls,
            )

    except ProviderUnavailable as exc:
        return AgentResult(text="", ok=False, error=str(exc), tool_calls=tool_calls)
    except MCPConnectionError as exc:
        return AgentResult(text="", ok=False, error=str(exc), tool_calls=tool_calls)
    except Exception as exc:  # noqa: BLE001 - última barreira: nada cru vaza
        # litellm levanta várias subclasses próprias; tratamos todas aqui.
        return AgentResult(
            text="",
            ok=False,
            error=f"Erro ao chamar o modelo ({type(exc).__name__}): {exc}",
            tool_calls=tool_calls,
        )
