"""Testes da seleção de modelo/provider para o LiteLLM.

Determinísticos: mockam os detectores de credencial e LLM_MODEL; sem rede.
"""

from __future__ import annotations

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
    assert provider._provider_of_model("claude-3-5-sonnet-latest") == "anthropic"
    assert provider._provider_of_model("gemini/gemini-1.5-pro") == "gemini"
    assert provider._provider_of_model("bedrock/anthropic.claude-x") == "bedrock"


def test_sem_nada_levanta_e_provider_available_false(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    _mock_creds(monkeypatch)
    with pytest.raises(provider.ProviderUnavailable):
        provider.select_model()
    assert provider.provider_available() is False
