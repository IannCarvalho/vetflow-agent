"""UI em Streamlit do assistente VetFlow.

Chat que roda o loop do agente Claude (agent.run_agent) sobre as ferramentas do
servidor vetflow-mcp. Cada chamada de ferramenta é mostrada "bonitinha" num
expander. A sidebar mostra o status da configuração -- NUNCA a senha nem a chave.

IMPORTANTE: `load_dotenv()` roda no TOPO, antes de importar `agent`, para que as
VETFLOW_* e a ANTHROPIC_API_KEY estejam no ambiente quando o agente subir o
subprocesso MCP (que herda `os.environ`).
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

# Carrega o .env ANTES de importar o agente/ponte, para que as variáveis cheguem
# também ao subprocesso MCP (que recebe uma cópia de os.environ).
load_dotenv()

import streamlit as st  # noqa: E402 - precisa vir após load_dotenv()

from agent import run_agent  # noqa: E402 - idem

st.set_page_config(page_title="Assistente VetFlow", page_icon="🐾", layout="centered")


def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _render_sidebar() -> None:
    """Mostra o status da configuração -- sem jamais revelar segredos."""
    with st.sidebar:
        st.header("Configuração")

        if _has_api_key():
            st.success("ANTHROPIC_API_KEY: definida")
        else:
            st.error("ANTHROPIC_API_KEY: ausente")

        base_url = os.environ.get("VETFLOW_BASE_URL", "(não definida)")
        email = os.environ.get("VETFLOW_EMAIL", "(não definido)")
        writes = os.environ.get("VETFLOW_ENABLE_WRITES", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        st.caption("Backend (via MCP)")
        st.text(f"URL: {base_url}")
        st.text(f"E-mail: {email}")
        st.text(f"Escritas: {'habilitadas' if writes else 'desabilitadas'}")
        # A senha NUNCA é exibida.

        st.divider()
        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()


def _render_tool_calls(tool_calls: list[dict]) -> None:
    """Renderiza cada ferramenta chamada num expander com nome + argumentos."""
    for call in tool_calls:
        with st.expander(f"🔧 Ferramenta: {call['name']}", expanded=False):
            st.caption("Argumentos")
            st.json(call.get("arguments") or {})


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # para render (role, text, tool_calls)
    if "history" not in st.session_state:
        st.session_state.history = []  # formato Anthropic (role/content)


def main() -> None:
    _init_state()
    st.title("🐾 Assistente VetFlow")
    st.caption("Converse em linguagem natural com os dados da clínica (via MCP + Claude).")

    _render_sidebar()

    # Replay do histórico já renderizado.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("tool_calls"):
                _render_tool_calls(msg["tool_calls"])
            if msg.get("text"):
                st.markdown(msg["text"])

    api_ok = _has_api_key()
    if not api_ok:
        st.warning(
            "A ANTHROPIC_API_KEY não está configurada. Defina-a no arquivo `.env` "
            "para habilitar o chat. A entrada está desabilitada até lá."
        )

    prompt = st.chat_input(
        "Pergunte algo sobre a clínica..." if api_ok else "Configure a ANTHROPIC_API_KEY para usar",
        disabled=not api_ok,
    )

    if not prompt:
        return

    # Mensagem do usuário.
    st.session_state.messages.append({"role": "user", "text": prompt, "tool_calls": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Resposta do agente.
    with st.chat_message("assistant"):
        placeholder = st.container()
        collected_calls: list[dict] = []

        def _on_tool_event(name: str, arguments: dict) -> None:
            collected_calls.append({"name": name, "arguments": arguments})
            with placeholder:
                with st.expander(f"🔧 Ferramenta: {name}", expanded=False):
                    st.caption("Argumentos")
                    st.json(arguments or {})

        try:
            result = asyncio.run(
                run_agent(
                    prompt,
                    history=st.session_state.history,
                    on_tool_event=_on_tool_event,
                )
            )
        except Exception as exc:  # noqa: BLE001 - última barreira da UI
            st.error(f"Erro inesperado: {exc}")
            return

        if not result.ok:
            st.error(result.error or "Ocorreu um erro desconhecido.")
            # Mantém o chat utilizável: registra a falha sem texto do assistente.
            st.session_state.messages.append(
                {"role": "assistant", "text": "", "tool_calls": collected_calls}
            )
            return

        if result.text:
            st.markdown(result.text)

        # Atualiza o histórico no formato Anthropic (user + assistant text).
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": result.text})
        st.session_state.messages.append(
            {"role": "assistant", "text": result.text, "tool_calls": collected_calls}
        )


if __name__ == "__main__":
    main()
