"""Seleção de modelo/provider para o LiteLLM, com fallback em cascata.

O LiteLLM expõe UMA interface (formato OpenAI) sobre muitos providers. Aqui só
decidimos QUAL `model` string passar e garantimos que exista credencial para
ele; o loop em agent.py não precisa saber de qual provider se trata.

Como o provider é escolhido:
1. Se `LLM_MODEL` estiver definido, ele manda (ex.: "gpt-4o",
   "claude-3-5-sonnet-latest", "gemini/gemini-1.5-pro",
   "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"). Validamos que a
   credencial do provider correspondente existe.
2. Senão, auto-detecção pela credencial presente, nesta ordem de preferência:
   Anthropic (ANTHROPIC_API_KEY) -> OpenAI (OPENAI_API_KEY) ->
   Gemini (GEMINI_API_KEY/GOOGLE_API_KEY) -> Bedrock (perfil .aws default).
3. Nada disponível -> ProviderUnavailable (chamador degrada com mensagem amigável).

Isto NÃO fala com nenhum provider; só inspeciona o ambiente e o `.aws`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Modelos default por provider (editáveis por env se o usuário quiser outro).
DEFAULT_ANTHROPIC = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
DEFAULT_OPENAI = os.environ.get("OPENAI_MODEL", "gpt-4o")
DEFAULT_GEMINI = os.environ.get("GEMINI_MODEL", "gemini/gemini-1.5-pro")
DEFAULT_BEDROCK = os.environ.get(
    "BEDROCK_MODEL_ID", "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
)


class ProviderUnavailable(RuntimeError):
    """Nenhum provider de LLM pôde ser configurado (nem chave, nem AWS)."""


@dataclass
class ModelSelection:
    """O que o agent.py precisa para chamar o LiteLLM."""

    model: str  # string de modelo no formato do LiteLLM
    provider: str  # rótulo legível: anthropic | openai | gemini | bedrock


# --- detectores de credencial (sem rede) ----------------------------------- #


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _has_gemini_key() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )


def _aws_credentials_available() -> bool:
    """True se o boto3 resolver credenciais AWS (perfil default, AWS_*, SSO...)."""
    try:
        import boto3
    except ImportError:
        return False
    try:
        session = boto3.session.Session(profile_name=os.environ.get("AWS_PROFILE") or None)
        return session.get_credentials() is not None
    except Exception:  # noqa: BLE001 - perfil inválido / sem credencial = indisponível
        return False


def _provider_of_model(model: str) -> str:
    """Infere o provider a partir da string de modelo do LiteLLM."""
    m = model.lower()
    if m.startswith("bedrock/"):
        return "bedrock"
    if m.startswith("gemini/") or m.startswith("google") or "gemini" in m:
        return "gemini"
    if m.startswith("gpt") or m.startswith("openai/") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    # claude-*, anthropic/* e o resto tratamos como Anthropic (API direta).
    return "anthropic"


def _credential_ok_for(provider: str) -> bool:
    return {
        "anthropic": _has_anthropic_key,
        "openai": _has_openai_key,
        "gemini": _has_gemini_key,
        "bedrock": _aws_credentials_available,
    }.get(provider, lambda: False)()


def select_model_with_overrides(
    model: str | None = None, api_key: str | None = None
) -> ModelSelection:
    """Resolve o modelo quando o REQUEST traz BYOK (chave do próprio usuário).

    - Se `model` + `api_key` vierem, usa-os direto (não consulta o ambiente): é
      o caminho BYOK, onde a chave vive no navegador do usuário e chega por
      request. A chave NÃO é guardada aqui -- quem a aplica é o agent.run_agent,
      passando-a ao litellm só naquela chamada.
    - Caso contrário, cai para a cascata baseada no ambiente (select_model).
    """
    m = (model or "").strip()
    k = (api_key or "").strip()
    if m and k:
        return ModelSelection(model=m, provider=_provider_of_model(m))
    if m and not k:
        # Modelo explícito do usuário, mas sem key no request: só é válido se o
        # ambiente tiver credencial para aquele provider (ex.: Bedrock via .aws).
        provider = _provider_of_model(m)
        if _credential_ok_for(provider):
            return ModelSelection(model=m, provider=provider)
    return select_model()


def select_model() -> ModelSelection:
    """Resolve o modelo a usar seguindo a cascata. Levanta ProviderUnavailable."""
    explicit = os.environ.get("LLM_MODEL", "").strip()
    if explicit:
        provider = _provider_of_model(explicit)
        if not _credential_ok_for(provider):
            raise ProviderUnavailable(
                f"LLM_MODEL='{explicit}' exige credenciais de '{provider}', que não "
                "foram encontradas. Defina a chave do provider (ou um perfil AWS "
                "para Bedrock)."
            )
        return ModelSelection(model=explicit, provider=provider)

    # Auto-detecção por credencial presente, por ordem de preferência.
    if _has_anthropic_key():
        return ModelSelection(DEFAULT_ANTHROPIC, "anthropic")
    if _has_openai_key():
        return ModelSelection(DEFAULT_OPENAI, "openai")
    if _has_gemini_key():
        return ModelSelection(DEFAULT_GEMINI, "gemini")
    if _aws_credentials_available():
        return ModelSelection(DEFAULT_BEDROCK, "bedrock")

    raise ProviderUnavailable(
        "Nenhum provider de LLM configurado. Defina uma destas no .env: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY -- ou configure um "
        "perfil AWS (`aws configure`) com acesso ao Amazon Bedrock."
    )


def provider_available() -> bool:
    """True se QUALQUER provider está utilizável agora."""
    try:
        select_model()
        return True
    except ProviderUnavailable:
        return False


def describe_provider() -> str:
    """Rótulo legível do provider/modelo que SERÁ usado -- sem revelar segredos."""
    try:
        sel = select_model()
    except ProviderUnavailable:
        return "nenhum provider configurado"
    if sel.provider == "bedrock":
        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "região padrão"
        )
        profile = os.environ.get("AWS_PROFILE", "default")
        return f"Amazon Bedrock ({sel.model}, perfil '{profile}', {region})"
    return f"{sel.provider} ({sel.model})"
