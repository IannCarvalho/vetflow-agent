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

# litellm é importado no load do módulo (startup do processo), NÃO dentro de
# run_agent: `import litellm` custa ~7s no 1o load e, feito numa coroutine,
# bloqueia o event loop por esse tempo — com 1 worker o /health trava e a
# plataforma reinicia o processo. Dependência de runtime obrigatória, então um
# ImportError aqui deve falhar o deploy (fail-fast), não virar erro por request.
import litellm

from .mcp_bridge import (
    MCPConnectionError,
    connect_session,
    dispatch_tool,
    list_tools,
    tools_to_openai,
)
from .provider import ProviderUnavailable, select_model_with_overrides

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


# --------------------------------------------------------------------------- #
# Fronteira de normalização da mensagem do LLM
#
# O LiteLLM devolve a mensagem do assistente ora como objeto (choices[0].message
# com .content/.tool_calls/.model_dump()), ora como dict -- a mesma quirk em dois
# formatos. Em vez de deixar esse "é dict ou objeto?" vazar por todo o loop de
# run_agent, concentramos a defensiva num seam só: `normalize_message` lê o cru e
# devolve um `AssistantMessage` tipado. O loop passa a ler `msg.text`,
# `msg.tool_calls[i].name/.arguments` e `msg.as_openai_dict()`, sem saber de qual
# formato o LiteLLM veio.
#
# Ganho: locality (a quirk mora num lugar) e superfície de teste (dá pra testar o
# parsing de tool-call -- inclusive arguments JSON inválido -- sem montar o loop).
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    """Uma chamada de ferramenta já normalizada.

    `arguments` já vem como dict: o json.loads/try-except (string JSON do provider,
    inválida -> {}) acontece dentro da normalização, não no loop.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantMessage:
    """Mensagem do assistente normalizada, independente do formato do LiteLLM."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Forma OpenAI original (dict) para anexar ao histórico sem perder campos.
    _raw_openai_dict: dict[str, Any] = field(default_factory=dict)

    def as_openai_dict(self) -> dict[str, Any]:
        """Reproduz o `message.model_dump()`/`dict(message)` para o histórico.

        É a mensagem do assistant (com os tool_calls) que o loop anexa a
        `messages` antes de despachar as ferramentas.
        """
        return self._raw_openai_dict


def _message_text(message: Any) -> str:
    """Extrai o texto da mensagem do assistente (dict ou objeto do LiteLLM)."""
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return (content or "").strip()


def _get_tool_calls(message: Any) -> list[Any]:
    """Lista de tool_calls crus da mensagem (vazia se não houver)."""
    if isinstance(message, dict):
        return message.get("tool_calls") or []
    return getattr(message, "tool_calls", None) or []


def _tc_field(tc: Any, *path: str) -> Any:
    """Acessa um campo de um tool_call cru, tolerante a dict vs objeto."""
    cur = tc
    for key in path:
        cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
        if cur is None:
            return None
    return cur


def _message_as_openai_dict(message: Any) -> dict[str, Any]:
    """Serializa a mensagem crua do assistente para dict no formato OpenAI."""
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return dict(message)


def _parse_arguments(raw_args: Any) -> dict[str, Any]:
    """Normaliza os arguments de um tool_call para dict (inválido -> {})."""
    if raw_args is None:
        raw_args = "{}"
    try:
        if isinstance(raw_args, str):
            return json.loads(raw_args)
        return dict(raw_args)
    except (ValueError, TypeError):
        return {}


def normalize_message(raw: Any) -> AssistantMessage:
    """Normaliza a mensagem crua do LiteLLM (dict OU objeto) em `AssistantMessage`.

    Este é o único ponto que conhece o formato cru do provider. Faz o parsing dos
    arguments de cada tool_call aqui dentro, para o loop lidar só com dados já
    tipados.
    """
    tool_calls = [
        ToolCall(
            id=_tc_field(tc, "id"),
            name=_tc_field(tc, "function", "name"),
            arguments=_parse_arguments(_tc_field(tc, "function", "arguments")),
        )
        for tc in _get_tool_calls(raw)
    ]
    return AssistantMessage(
        text=_message_text(raw),
        tool_calls=tool_calls,
        _raw_openai_dict=_message_as_openai_dict(raw),
    )


async def run_agent(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    on_tool_event: Callable[[str, dict[str, Any]], None] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    user_token: str | None = None,
) -> AgentResult:
    """Roda uma rodada do agente para `user_message` (nunca levanta).

    `model`/`api_key` são o caminho BYOK: quando vêm do request (chave do
    navegador do usuário), são usados só nesta chamada e NUNCA persistidos.
    `user_token` é o JWT do usuário logado (pass-through): repassado ao MCP para
    que ele aja como esse usuário, em vez de uma conta de serviço.
    """
    # Fail-fast de provider ANTES de abrir a sessão MCP: sem credencial de LLM
    # não há como responder, então resolver o provider primeiro evita levantar o
    # MCP à toa. Checagem barata (só inspeciona o ambiente/.aws, não fala com
    # provider nenhum). O custo pesado — `import litellm`, ~7s no 1o load — já
    # foi pago no load do módulo (ver import no topo), e NÃO aqui dentro da
    # coroutine, para não bloquear o event loop (causa do restart do worker no
    # plano free: /health trava e o Render reinicia).
    try:
        selection = select_model_with_overrides(model=model, api_key=api_key)
    except ProviderUnavailable as exc:
        return AgentResult(
            text="",
            ok=False,
            error=(
                "Nenhum provider de LLM configurado. Informe uma API key no chat "
                f"(BYOK) ou configure o servidor. Detalhe: {exc}"
            ),
        )

    tool_calls: list[dict[str, Any]] = []

    try:
        chosen_model = selection.model
        # Chave BYOK (se veio no request) é passada ao litellm só nesta chamada.
        byok_key = (api_key or "").strip() or None

        async with connect_session(user_token=user_token) as session:
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
                msg = normalize_message(response.choices[0].message)

                if not msg.tool_calls:
                    return AgentResult(
                        text=msg.text,
                        ok=True,
                        tool_calls=tool_calls,
                    )

                # Anexa a mensagem do assistente (com os tool_calls) e despacha.
                messages.append(msg.as_openai_dict())

                for tc in msg.tool_calls:
                    if on_tool_event is not None:
                        on_tool_event(tc.name, tc.arguments)
                    tool_calls.append({"name": tc.name, "arguments": tc.arguments})

                    text, _is_error = await dispatch_tool(session, tc.name, tc.arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
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
