"""Testes da seleção de modelo/provider para o LiteLLM.

Determinísticos: mockam os detectores de credencial e LLM_MODEL; sem rede. Os
canários consultam só o registro local de modelos do litellm.
"""

from __future__ import annotations

import litellm
import pytest

import vetflow_agent.core.provider as provider


def _mock_creds(monkeypatch, *, anthropic=False, openai=False, gemini=False, aws=False):
    monkeypatch.setattr(provider, "_has_anthropic_key", lambda: anthropic)
    monkeypatch.setattr(provider, "_has_openai_key", lambda: openai)
    monkeypatch.setattr(provider, "_has_gemini_key", lambda: gemini)
    monkeypatch.setattr(provider, "_aws_credentials_available", lambda: aws)


def test_auto_prefere_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    _mock_creds(monkeypatch, anthropic=True, openai=True)
    sel = provider.select_model()
    assert sel.provider == "anthropic"
    assert sel.model == provider.DEFAULT_ANTHROPIC


def test_auto_cai_para_openai_sem_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    _mock_creds(monkeypatch, openai=True)
    sel = provider.select_model()
    assert sel.provider == "openai"


def test_auto_cai_para_gemini(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    _mock_creds(monkeypatch, gemini=True)
    assert provider.select_model().provider == "gemini"


def test_auto_cai_para_bedrock_por_ultimo(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    _mock_creds(monkeypatch, aws=True)
    assert provider.select_model().provider == "bedrock"


def test_llm_model_explicito_manda(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    _mock_creds(monkeypatch, anthropic=True, openai=True)  # chave p/ openai existe
    sel = provider.select_model()
    assert sel.provider == "openai"
    assert sel.model == "gpt-4o"


def test_llm_model_explicito_sem_credencial_levanta(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    _mock_creds(monkeypatch, anthropic=True)  # só anthropic, mas pediu openai
    with pytest.raises(provider.ProviderUnavailable):
        provider.select_model()


def test_provider_of_model_infere_corretamente():
    assert provider._provider_of_model("gpt-4o") == "openai"
    assert provider._provider_of_model("gpt-6-luna") == "openai"
    assert provider._provider_of_model("openai/gpt-6-luna") == "openai"
    assert provider._provider_of_model("claude-sonnet-5-5") == "anthropic"
    assert provider._provider_of_model("anthropic/claude-sonnet-5-5") == "anthropic"
    assert provider._provider_of_model("gemini/gemini-3.8-flash") == "gemini"
    assert provider._provider_of_model("bedrock/anthropic.claude-x") == "bedrock"


@pytest.mark.parametrize("model", ["o1", "o3", "o3-mini", "o4-mini", "O4-MINI"])
def test_provider_of_model_familia_o_e_openai(model):
    # Regressão: só o1/o3 eram reconhecidos; o4-mini caía no fallback anthropic.
    assert provider._provider_of_model(model) == "openai"


def test_llm_model_o4_mini_usa_a_credencial_da_openai(monkeypatch):
    # Antes: "LLM_MODEL='o4-mini' exige credenciais de 'anthropic'".
    monkeypatch.setenv("LLM_MODEL", "o4-mini")
    _mock_creds(monkeypatch, openai=True)
    sel = provider.select_model()
    assert (sel.provider, sel.model) == ("openai", "o4-mini")


# Canários contra o registro do litellm instalado. O CI instala a versão mais
# recente, então, quando um modelo default for aposentado e sair do registro,
# estes testes falham e avisam que é hora de trocar o default.
_DEFAULTS = [
    ("anthropic", provider.DEFAULT_ANTHROPIC),
    ("openai", provider.DEFAULT_OPENAI),
    ("gemini", provider.DEFAULT_GEMINI),
    ("bedrock", provider.DEFAULT_BEDROCK),
]


@pytest.mark.parametrize(("esperado", "model"), _DEFAULTS)
def test_defaults_sao_reconhecidos_pelo_litellm(esperado, model):
    _, inferido, _, _ = litellm.get_llm_provider(model)
    assert inferido == esperado
    # O loop do agente depende de tool calling.
    assert litellm.supports_function_calling(model=model)


@pytest.mark.parametrize(
    "model",
    [m for _, m in _DEFAULTS]
    + ["gpt-4o", "o3", "o4-mini", "openai/gpt-6-luna", "anthropic/claude-sonnet-5-5"],
)
def test_provider_of_model_concorda_com_o_litellm(model):
    _, inferido, _, _ = litellm.get_llm_provider(model)
    assert provider._provider_of_model(model) == inferido


def test_sem_nada_levanta_e_provider_available_false(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    _mock_creds(monkeypatch)
    with pytest.raises(provider.ProviderUnavailable):
        provider.select_model()
    assert provider.provider_available() is False
