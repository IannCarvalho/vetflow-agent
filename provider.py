"""Seleção de provider do Claude, com fallback em cascata.

Decide COMO autenticar no Claude, sem o agent.py precisar saber dos detalhes:

1. ANTHROPIC_API_KEY (API direta da Anthropic) -- se a variável existir no
   ambiente (vinda do .env OU já configurada na máquina/sistema). O SDK também
   lê ANTHROPIC_API_KEY sozinho; aqui tornamos a decisão explícita.
2. Amazon Bedrock -- se não houver chave Anthropic mas houver credenciais AWS
   resolvíveis (perfil `default` do ~/.aws, ou AWS_PROFILE/variáveis AWS_*). O
   AsyncAnthropicBedrock usa a cadeia padrão do boto3, então o `.aws default` é
   consumido automaticamente quando nada é declarado.
3. Nenhum -> ProviderUnavailable, para o chamador degradar com mensagem amigável.

Model ids diferem por provider: a API direta aceita apelidos (`-latest`); o
Bedrock exige o id versionado com prefixo `anthropic.`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Modelos equivalentes por provider (Claude 3.5 Sonnet).
ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
BEDROCK_MODEL = os.environ.get(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
)


class ProviderUnavailable(RuntimeError):
    """Nenhum provider do Claude pôde ser configurado (nem chave, nem AWS)."""


@dataclass
class ClientBundle:
    """Cliente do Claude já instanciado + metadados para o loop do agente."""

    client: Any
    model: str
    provider: str  # "anthropic" | "bedrock" -- rótulo para a UI/logs


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _aws_credentials_available() -> bool:
    """True se o boto3 conseguir resolver credenciais AWS (perfil default etc.).

    Não faz chamada de rede: só pergunta à cadeia de credenciais do botocore se
    há algo resolvível (perfil `default` do ~/.aws, variáveis AWS_*, SSO, etc.).
    """
    try:
        import boto3
    except ImportError:
        return False
    try:
        session = boto3.session.Session(
            profile_name=os.environ.get("AWS_PROFILE") or None
        )
        return session.get_credentials() is not None
    except Exception:  # noqa: BLE001 - perfil inválido / sem credencial = indisponível
        return False


def _aws_region() -> str | None:
    """Região AWS a usar no Bedrock, do ambiente ou do perfil configurado."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        return region
    try:
        import boto3

        return boto3.session.Session(
            profile_name=os.environ.get("AWS_PROFILE") or None
        ).region_name
    except Exception:  # noqa: BLE001
        return None


def build_client() -> ClientBundle:
    """Monta o cliente do Claude seguindo a cascata de providers.

    Levanta ProviderUnavailable se nenhum caminho estiver disponível.
    """
    import anthropic

    # 1) API direta da Anthropic (chave do .env ou já no ambiente da máquina).
    if _has_anthropic_key():
        return ClientBundle(
            client=anthropic.AsyncAnthropic(),  # lê ANTHROPIC_API_KEY do ambiente
            model=ANTHROPIC_MODEL,
            provider="anthropic",
        )

    # 2) Bedrock via credenciais AWS resolvíveis (perfil default do ~/.aws).
    if _aws_credentials_available():
        kwargs: dict[str, Any] = {}
        region = _aws_region()
        if region:
            kwargs["aws_region"] = region
        if os.environ.get("AWS_PROFILE"):
            kwargs["aws_profile"] = os.environ["AWS_PROFILE"]
        return ClientBundle(
            client=anthropic.AsyncAnthropicBedrock(**kwargs),
            model=BEDROCK_MODEL,
            provider="bedrock",
        )

    # 3) Nada disponível.
    raise ProviderUnavailable(
        "Nenhuma credencial do Claude encontrada. Defina ANTHROPIC_API_KEY no "
        ".env, ou configure um perfil AWS (ex.: `aws configure`) com acesso ao "
        "Amazon Bedrock."
    )


def provider_available() -> bool:
    """True se ALGUM provider do Claude está configurado (chave OU AWS)."""
    return _has_anthropic_key() or _aws_credentials_available()


def describe_provider() -> str:
    """Rótulo legível do provider que SERÁ usado -- para a sidebar, sem segredos.

    Não instancia cliente nem revela credenciais; só diz qual caminho está
    disponível agora.
    """
    if _has_anthropic_key():
        return "Anthropic API (ANTHROPIC_API_KEY)"
    if _aws_credentials_available():
        region = _aws_region() or "região padrão"
        profile = os.environ.get("AWS_PROFILE", "default")
        return f"Amazon Bedrock (perfil AWS '{profile}', {region})"
    return "nenhum provider configurado"
