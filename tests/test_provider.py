"""Testes da seleção de provider (cascata chave -> bedrock -> indisponível).

Determinísticos: mockam os detectores (_has_anthropic_key, _aws_credentials_available)
e o SDK anthropic, sem tocar em credenciais reais nem na rede.
"""

from __future__ import annotations

import sys
import types

import pytest

import provider


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Instala um módulo anthropic falso com os dois clientes."""
    mod = types.ModuleType("anthropic")

    class _Client:
        def __init__(self, *a, **k):
            self.args = a
            self.kwargs = k

    mod.AsyncAnthropic = _Client
    mod.AsyncAnthropicBedrock = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return mod


def test_build_client_prefere_anthropic_quando_ha_chave(monkeypatch, fake_anthropic):
    monkeypatch.setattr(provider, "_has_anthropic_key", lambda: True)
    monkeypatch.setattr(provider, "_aws_credentials_available", lambda: True)

    bundle = provider.build_client()

    assert bundle.provider == "anthropic"
    assert bundle.model == provider.ANTHROPIC_MODEL


def test_build_client_cai_para_bedrock_sem_chave(monkeypatch, fake_anthropic):
    monkeypatch.setattr(provider, "_has_anthropic_key", lambda: False)
    monkeypatch.setattr(provider, "_aws_credentials_available", lambda: True)
    monkeypatch.setattr(provider, "_aws_region", lambda: "us-east-1")

    bundle = provider.build_client()

    assert bundle.provider == "bedrock"
    assert bundle.model == provider.BEDROCK_MODEL


def test_build_client_sem_nada_levanta_provider_unavailable(monkeypatch, fake_anthropic):
    monkeypatch.setattr(provider, "_has_anthropic_key", lambda: False)
    monkeypatch.setattr(provider, "_aws_credentials_available", lambda: False)

    with pytest.raises(provider.ProviderUnavailable):
        provider.build_client()


def test_provider_available_reflete_qualquer_caminho(monkeypatch):
    monkeypatch.setattr(provider, "_has_anthropic_key", lambda: False)
    monkeypatch.setattr(provider, "_aws_credentials_available", lambda: False)
    assert provider.provider_available() is False

    monkeypatch.setattr(provider, "_aws_credentials_available", lambda: True)
    assert provider.provider_available() is True
