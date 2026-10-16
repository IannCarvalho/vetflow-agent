"""Ponte entre o app Streamlit e o servidor `vetflow-mcp`.

Sobe o `vetflow-mcp` como SUBPROCESSO via stdio (o transporte padrão do SDK
`mcp`), descobre as tools expostas em tempo de execução, converte cada uma para
o formato de ferramenta da Anthropic e despacha chamadas. Toda falha é
traduzida para algo legível em PT-BR -- nenhum traceback cru escapa.

Decisão de design: o app NÃO fala com o backend Go. Ele só conhece o MCP. As
credenciais `VETFLOW_*` são passadas para o processo filho via `env`, e é o MCP
quem autentica no backend. Assim esta ponte é um módulo fino: ligar, listar,
despachar, desligar.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Raiz do projeto (onde fica o .env em dev). Usada como cwd do subprocesso MCP
# para que ele ache o .env e o pacote vetflow_mcp instalado no mesmo venv.
# Layout src: este módulo é src/vetflow_agent/core/mcp_bridge.py, então a raiz
# está 3 níveis acima (core -> vetflow_agent -> src -> raiz do projeto).
APP_DIR = Path(__file__).resolve().parents[3]


class MCPConnectionError(RuntimeError):
    """Falha ao conectar/inicializar o servidor MCP, com mensagem amigável.

    Guarda também a causa concreta (`detail`) extraída do stderr do filho ou da
    folha de um ExceptionGroup do anyio, quando disponível.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


def _check_required_env(user_token: str | None = None) -> None:
    """Levanta MCPConnectionError amigável se faltar credencial obrigatória.

    Pass-through: se um `user_token` (JWT do usuário logado) for fornecido, ou
    `VETFLOW_TOKEN` estiver no ambiente, então email/senha são opcionais —
    basta a base_url. Senão, exige email+senha (modo conta de serviço).
    """
    has_token = bool((user_token or "").strip() or os.environ.get("VETFLOW_TOKEN", "").strip())
    missing: list[str] = []
    if not os.environ.get("VETFLOW_BASE_URL", "").strip():
        missing.append("VETFLOW_BASE_URL")
    if not has_token:
        for v in ("VETFLOW_EMAIL", "VETFLOW_PASSWORD"):
            if not os.environ.get(v, "").strip():
                missing.append(v)
    if missing:
        detail = "Variáveis de ambiente obrigatórias ausentes: " + ", ".join(missing)
        raise MCPConnectionError(
            "Não consigo conversar com o VetFlow: faltam credenciais de conexão. "
            f"{detail}. Preencha-as no arquivo .env.",
            detail=detail,
        )


def _server_params(user_token: str | None = None) -> StdioServerParameters:
    """Parâmetros para lançar o `vetflow-mcp` como subprocesso stdio.

    Repassa uma CÓPIA do ambiente atual para que as VETFLOW_* (carregadas pelo
    app via load_dotenv) cheguem ao filho. Se `user_token` for fornecido (JWT do
    usuário logado), injeta `VETFLOW_TOKEN` no env do filho — o MCP então age
    como esse usuário (pass-through), em vez de logar com uma conta de serviço.
    O stderr do filho é redirecionado pelo `stdio_client` via `errlog`.
    """
    child_env = dict(os.environ)  # cópia; inclui as VETFLOW_* já no ambiente
    if (user_token or "").strip():
        child_env["VETFLOW_TOKEN"] = user_token.strip()
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "vetflow_mcp.server"],
        env=child_env,
        cwd=str(APP_DIR),
    )


def _unwrap_exception(exc: BaseException) -> BaseException:
    """Desembrulha ExceptionGroup do anyio até a folha concreta.

    O anyio embrulha a causa real num ExceptionGroup cujo str() é apenas
    "unhandled errors in a TaskGroup". Descemos recursivamente pelo primeiro
    membro de `.exceptions` até achar algo que não seja um grupo.
    """
    seen: set[int] = set()
    current: BaseException = exc
    while True:
        if id(current) in seen:
            return current
        seen.add(id(current))
        sub = getattr(current, "exceptions", None)
        if sub:
            current = sub[0]
            continue
        return current


def _friendly_detail(leaf: BaseException, stderr_text: str) -> str:
    """Extrai a melhor explicação possível para uma falha de conexão.

    Prioriza o stderr do filho (onde o MCP imprime o SystemExit das credenciais
    ausentes); cai para a str da folha do ExceptionGroup.
    """
    text = (stderr_text or "").strip()
    if text:
        # A última linha não-vazia costuma ser a mensagem do SystemExit.
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line:
                return line
    leaf_str = str(leaf).strip()
    return leaf_str or leaf.__class__.__name__


def _reraise_body_error(exc: Exception) -> NoReturn:
    """Repropaga um erro do CORPO do `async with connect_session()`.

    Um erro levantado por quem USA a sessão (ex.: o LLM recusou a API key) não é
    falha de conexão com o MCP e não pode virar MCPConnectionError. Antes, a
    mensagem mandava o usuário conferir as VETFLOW_* e o backend quando o
    problema era a key. No modo subprocesso o anyio (stdio_client/ClientSession)
    embrulha a exceção em ExceptionGroups aninhados, então desembrulhamos para o
    chamador receber o tipo real (ex.: litellm.AuthenticationError). No modo
    in-process ela já chega crua e sobe intacta.
    """
    leaf = _unwrap_exception(exc)
    if leaf is exc:
        raise exc
    raise leaf from None


@asynccontextmanager
async def connect_session(user_token: str | None = None):
    """Context manager assíncrono que cede uma ClientSession inicializada.

    Uso:
        async with connect_session() as session:
            tools = await list_tools(session)
            ...

    Dois modos, selecionados por `VETFLOW_MCP_INPROCESS`:

    - **in-process** (flag ligado): importa o servidor `vetflow_mcp` e chama
      suas tools no MESMO processo — sem subir um segundo interpretador Python.
      Elimina o pico de RAM do subprocesso (causa do OOM no plano free). O token
      do usuário é escopado por request via `use_request_client` (contextvars),
      então o isolamento de multi-tenancy é preservado mesmo sem processos
      separados. Cede um adaptador que expõe a MESMA interface da ClientSession
      (`list_tools`/`call_tool`), para que `agent.py` não precise mudar.

    - **subprocesso** (default): sobe o `vetflow-mcp` como subprocesso stdio.
      Modo histórico, idêntico ao do Claude Desktop. Mais isolado (um crash do
      MCP não derruba a API), ao custo de um segundo interpretador residente.

    Falhas da CONEXÃO (abrir, inicializar ou fechar a sessão) viram
    `MCPConnectionError` com mensagem amigável em PT-BR. Um erro levantado por
    quem USA a sessão (ex.: o LLM recusou a key) sobe com o tipo original, e o
    cancelamento propaga normalmente.
    """
    if _inprocess_enabled():
        async with _connect_inprocess(user_token) as session:
            yield session
        return

    async with _connect_subprocess(user_token) as session:
        yield session


def _inprocess_enabled() -> bool:
    """True se o consumo in-process do MCP estiver ligado por env."""
    return os.environ.get("VETFLOW_MCP_INPROCESS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class _InProcessSession:
    """Adaptador que faz um `MCPServer` in-process parecer uma `ClientSession`.

    Expõe só o que o bridge usa — `list_tools()` (devolvendo um objeto com
    `.tools`, como o protocolo) e `call_tool(name, arguments=...)` (devolvendo o
    `CallToolResult` que `_flatten_tool_result` já sabe achatar). Assim os
    helpers `list_tools`/`dispatch_tool` e o `agent.py` funcionam sem mudança.
    """

    def __init__(self, server: Any) -> None:
        self._server = server

    async def list_tools(self) -> Any:
        tools = await self._server.list_tools()
        return SimpleNamespace(tools=list(tools))

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return await self._server.call_tool(name, arguments or {})


@asynccontextmanager
async def _connect_inprocess(user_token: str | None = None):
    """Cede uma sessão in-process, com o client escopado ao token da request."""
    _check_required_env(user_token)
    try:
        from vetflow_mcp import server as mcp_server
    except ImportError as exc:  # pacote não instalado no venv
        raise MCPConnectionError(
            "Modo in-process pedido (VETFLOW_MCP_INPROCESS) mas o pacote "
            "vetflow_mcp não está instalado. Rode: pip install -e ../vetflow-mcp "
            "(ou pip install ./vendor/vetflow-mcp).",
            detail=str(exc),
        ) from exc

    # Só falhas da CONEXÃO (abrir/fechar o client) viram MCPConnectionError.
    # `body_running` marca o trecho em que quem está rodando é o chamador: um
    # erro ali é dele e sobe como está. `except Exception` (não BaseException)
    # deixa o cancelamento (CancelledError) propagar em vez de virar erro.
    body_running = False
    try:
        # use_request_client escopa o VetFlowClient (com o JWT do usuário) ao
        # contexto desta request via contextvars — isolamento por-usuário sem
        # subir processo. As tools do server resolvem esse client.
        async with mcp_server.use_request_client(token=user_token):
            body_running = True
            yield _InProcessSession(mcp_server.mcp)
            body_running = False
    except MCPConnectionError:
        raise
    except Exception as exc:
        if body_running:
            _reraise_body_error(exc)
        leaf = _unwrap_exception(exc)
        detail = _friendly_detail(leaf, "")
        raise MCPConnectionError(
            "Falha ao consumir o vetflow-mcp in-process. Verifique as variáveis "
            f"VETFLOW_* e se o backend VetFlow está no ar. Causa: {detail}",
            detail=detail,
        ) from leaf


@asynccontextmanager
async def _connect_subprocess(user_token: str | None = None):
    """Sobe o `vetflow-mcp` como subprocesso stdio e cede a ClientSession.

    Modo histórico (default). Traduz uma saída precoce do filho (credencial
    ausente, backend fora) em `MCPConnectionError` amigável, usando o stderr
    capturado + a folha do ExceptionGroup do anyio.
    """
    # Falha cedo e amigável se faltar credencial obrigatória (determinístico,
    # independente do comportamento do `-m` no Windows).
    _check_required_env(user_token)

    # Arquivo temporário real para o stderr do filho (precisa de fileno).
    errlog = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    # Mesma regra do in-process: só a conexão vira MCPConnectionError; erro do
    # corpo é do chamador, e o cancelamento propaga.
    body_running = False
    try:
        try:
            async with stdio_client(_server_params(user_token), errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    body_running = True
                    yield session
                    body_running = False
        except MCPConnectionError:
            raise
        except Exception as exc:
            if body_running:
                _reraise_body_error(exc)
            leaf = _unwrap_exception(exc)
            try:
                errlog.seek(0)
                stderr_text = errlog.read()
            except Exception:  # noqa: BLE001 - leitura de diagnóstico é best-effort
                stderr_text = ""
            detail = _friendly_detail(leaf, stderr_text)
            raise MCPConnectionError(
                "Não consegui iniciar o servidor vetflow-mcp. Verifique as "
                "variáveis VETFLOW_* no .env e se o backend VetFlow está no ar. "
                f"Causa: {detail}",
                detail=detail,
            ) from leaf
    finally:
        with suppress(Exception):
            errlog.close()


async def list_tools(session: ClientSession) -> list[Any]:
    """Lista as tools expostas pelo servidor MCP (objetos Tool do SDK).

    Falhar aqui é falha de CONEXÃO (o servidor caiu entre o initialize e a
    listagem), então vira MCPConnectionError, e não erro do modelo.
    """
    try:
        result = await session.list_tools()
    except Exception as exc:
        leaf = _unwrap_exception(exc)
        detail = _friendly_detail(leaf, "")
        raise MCPConnectionError(
            f"Não consegui listar as ferramentas do vetflow-mcp. Causa: {detail}",
            detail=detail,
        ) from leaf
    return list(result.tools)


def tool_to_openai(tool: Any) -> dict[str, Any]:
    """Converte uma Tool MCP para o formato de function-calling da OpenAI.

    É o formato que o LiteLLM espera (e traduz para cada provider):
    {"type": "function", "function": {"name", "description", "parameters"}}.
    """
    schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "").strip(),
            "parameters": schema,
        },
    }


def tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
    """Converte uma lista de Tools MCP para o formato OpenAI/LiteLLM."""
    return [tool_to_openai(t) for t in tools]


def _flatten_tool_result(result: Any) -> str:
    """Achata um CallToolResult concatenando o `.text` dos blocos TextContent."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    # Fallback: alguns servidores devolvem apenas structured_content.
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return str(structured)
    return ""


async def dispatch_tool(
    session: ClientSession, name: str, arguments: dict[str, Any]
) -> tuple[str, bool]:
    """Executa uma tool MCP e devolve (texto, is_error).

    NUNCA levanta: qualquer exceção vira (mensagem_amigável, True), para que o
    loop do agente possa repassar o erro ao modelo em vez de quebrar.
    """
    try:
        result = await session.call_tool(name, arguments=arguments or {})
    except Exception as exc:  # noqa: BLE001 - erro de tool nunca deve derrubar o loop
        leaf = _unwrap_exception(exc)
        return (
            f"Erro ao executar a ferramenta '{name}': {leaf}",
            True,
        )

    is_error = bool(getattr(result, "is_error", False))
    text = _flatten_tool_result(result)
    if not text:
        text = "(a ferramenta não retornou conteúdo textual)"
    return text, is_error
