"""Configuração de testes do vetflow-agent.

Instala um stub mínimo do pacote `anthropic` em sys.modules, já que
`agent.run_agent` faz `import anthropic` por dentro. O stub expõe exatamente o
que o código toca: AsyncAnthropic e AnthropicError. Testes que precisam dirigir
o loop de tool-use substituem AsyncAnthropic por um fake via monkeypatch no
atributo do próprio módulo stub.

O pacote `vetflow_agent` é importável porque é instalado em modo editável
(`pip install -e .`), então não há manipulação de sys.path aqui.
"""

from __future__ import annotations

import sys
import types


def _install_anthropic_stub() -> None:
    """Instala um módulo `anthropic` mínimo em sys.modules se não houver um real.

    agent.py faz `import anthropic` dentro de run_agent e referencia
    anthropic.AsyncAnthropic e anthropic.AnthropicError. Um stub leve basta para
    os testes determinísticos; se o pacote real estiver instalado, não mexemos.
    """
    if "anthropic" in sys.modules:
        return
    try:  # pragma: no cover - caminho só exercitado quando o real existe
        import anthropic  # noqa: F401

        return
    except Exception:  # noqa: BLE001 - sem o real, instalamos o stub
        pass

    stub = types.ModuleType("anthropic")

    class AnthropicError(Exception):
        """Erro base da Anthropic (stub)."""

    class AsyncAnthropic:  # pragma: no cover - substituído por fake nos testes
        def __init__(self, *args, **kwargs) -> None:
            self.messages = None

    stub.AnthropicError = AnthropicError
    stub.AsyncAnthropic = AsyncAnthropic
    sys.modules["anthropic"] = stub


_install_anthropic_stub()
