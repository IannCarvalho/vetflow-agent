"""Smoke test da UI Streamlit (src/vetflow_agent/ui/app.py) com o AppTest.

Roda o script sem navegador e sem mensagem do usuário, então nada chega ao MCP
nem ao LLM. Pega erro de import, parâmetro do Streamlit que deixou de existir e
texto errado na tela. Pulado quando o streamlit não está instalado: ele fica em
requirements-dev.txt, fora do deploy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Paga o `import litellm` (~7s no 1o load) antes do AppTest, cujo timeout é por run.
import vetflow_agent.core.agent  # noqa: F401
import vetflow_agent.core.provider as provider

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "vetflow_agent" / "ui" / "app.py"


def _run(monkeypatch, *, disponivel: bool):
    # O app.py faz `from ...provider import ...` ao rodar, então pega estes fakes.
    monkeypatch.setattr(provider, "provider_available", lambda: disponivel)
    monkeypatch.setattr(provider, "describe_provider", lambda: "openai (gpt-6-luna)")
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_ui_sem_provider_avisa_e_desabilita_o_chat(monkeypatch):
    at = _run(monkeypatch, disponivel=False)
    assert at.sidebar.error[0].value == "Nenhum provider de LLM configurado"
    assert "Nenhum provider de LLM" in at.warning[0].value
    assert at.chat_input[0].disabled


def test_ui_com_provider_mostra_o_llm_e_libera_o_chat(monkeypatch):
    at = _run(monkeypatch, disponivel=True)
    assert at.sidebar.success[0].value == "LLM: openai (gpt-6-luna)"
    assert not at.warning
    assert not at.chat_input[0].disabled
