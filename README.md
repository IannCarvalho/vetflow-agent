# 🐾 vetflow-agent

[![CI](https://github.com/IannCarvalho/vetflow-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/IannCarvalho/vetflow-agent/actions/workflows/ci.yml)

Interface de **chat em Streamlit** que roda um **agente da Anthropic (Claude)**
sobre as ferramentas expostas pelo servidor MCP [`vetflow-mcp`](../vetflow-mcp).
Você conversa em linguagem natural ("quais vacinas vencem esse mês?", "quantos
pets a Maria tem?") e o Claude usa as ferramentas do MCP para consultar os dados
reais da clínica antes de responder.

## O que é (Opção B: app próprio)

Em vez de plugar o `vetflow-mcp` num cliente MCP pronto (Claude Desktop), este é
um **app autocontido** que embute o loop do agente. Isso dá controle total sobre
a UI, o system prompt e a forma como as chamadas de ferramenta são exibidas --
cada tool que o Claude dispara aparece "bonitinha" num expander com nome e
argumentos.

> Este projeto **não modifica** o `vetflow-mcp`. Apenas o consome.

## Como conecta (arquitetura)

```
  Você (navegador)
        │
        ▼
   app.py (Streamlit)
        │  asyncio.run(run_agent(...))
        ▼
   agent.py ── loop de tool-use (formato OpenAI) ──► LiteLLM ──► Claude / GPT / Gemini / Bedrock
        │                                                           ▲
        │ dispatch_tool / list_tools                                │ tools (auto-discovery)
        ▼                                                           │
   mcp_bridge.py ── stdio (subprocesso) ──► vetflow-mcp (python -m vetflow_mcp.server)
                                                   │  Bearer JWT
                                                   ▼
                                           Backend VetFlow (Go) ──► MongoDB
```

- **`mcp_bridge.py`** sobe o `vetflow-mcp` como **subprocesso via stdio** (o
  transporte padrão do SDK `mcp`), descobre as tools em tempo de execução
  (*auto-discovery*) e as converte para o formato de function-calling da OpenAI
  (que o LiteLLM usa). As credenciais `VETFLOW_*` são repassadas ao filho pelo
  ambiente; o app nunca fala com o backend Go diretamente.
- **`agent.py`** implementa o **loop de tool-use** no formato OpenAI, via
  `litellm.acompletion`: pede uma resposta ao modelo, executa cada `tool_call`,
  devolve uma mensagem `role:"tool"` casada por `tool_call_id` e repete até o
  modelo parar de pedir ferramentas (trava `MAX_TURNS = 12`).
- **`provider.py`** decide **qual LLM usar** (multi-provider via LiteLLM): honra
  `LLM_MODEL` se definido, senão auto-detecta pela credencial presente nesta
  ordem — Anthropic → OpenAI → Gemini → **Bedrock (perfil `.aws` default)**.
- **`app.py`** é a UI: chat, histórico, exibição das ferramentas e uma sidebar
  de status que mostra o provider ativo e **nunca mostra senha nem chave**.

## Multi-provider (LiteLLM)

O agente roda sobre **qualquer LLM** suportado pelo LiteLLM. Você escolhe de
duas formas:

- **Explícito:** defina `LLM_MODEL` no `.env` — ex. `gpt-4o`,
  `claude-3-5-sonnet-latest`, `gemini/gemini-1.5-pro`,
  `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`.
- **Automático:** não defina `LLM_MODEL`; o app usa o primeiro provider com
  credencial disponível, nesta ordem de preferência: `ANTHROPIC_API_KEY` →
  `OPENAI_API_KEY` → `GEMINI_API_KEY` → **Amazon Bedrock** (credenciais do
  perfil `.aws` default, sem precisar declarar nada).

## Abordagem de integração escolhida

O `vetflow-mcp` é instalado em **modo editável no mesmo venv**:

```bash
pip install -e ../vetflow-mcp
```

Assim o subprocesso (`python -m vetflow_mcp.server`) roda com o **mesmo
interpretador** do app (`sys.executable`) e encontra o pacote `vetflow_mcp`.
Nada é duplicado: você edita o MCP no seu próprio repositório e o agente passa a
enxergar as mudanças na próxima execução.

## Setup

Requer **Python >= 3.10** (ideal 3.12).

```bash
# a partir de vetflow-agent/
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e ../vetflow-mcp      # instala o servidor MCP no mesmo venv

cp .env.example .env               # (ou copie manualmente no Windows)
# edite o .env: ANTHROPIC_API_KEY + VETFLOW_BASE_URL/EMAIL/PASSWORD
```

Rode:

```bash
streamlit run app.py
```

## Variáveis de ambiente

| Variável | Papel |
|---|---|
| `ANTHROPIC_API_KEY` | Chave do Claude. Sem ela, o chat fica desabilitado (aviso amigável). |
| `VETFLOW_BASE_URL` | URL do backend VetFlow (ex.: `http://localhost:8080`). |
| `VETFLOW_EMAIL` / `VETFLOW_PASSWORD` | Login do vet; o MCP autentica no backend. |
| `VETFLOW_TIMEOUT_S` | Timeout HTTP em segundos (default 15). |
| `VETFLOW_ENABLE_WRITES` | Liga as tools de escrita do MCP (default `false`). |

Todas vêm do `.env` (carregado no topo do `app.py`, antes de importar o agente,
para que cheguem também ao subprocesso MCP).

## Degradação graciosa (sem backend / sem chave)

O app foi feito para **não quebrar** quando falta configuração:

- **Sem `ANTHROPIC_API_KEY`:** a entrada de chat é desabilitada com um aviso; a
  UI continua navegável.
- **Sem as `VETFLOW_*` / backend fora do ar:** o subprocesso MCP sai cedo
  (`SystemExit`). A ponte captura o `stderr` do filho e desembrulha o
  `ExceptionGroup` do anyio até a causa real, traduzindo tudo para uma mensagem
  amigável em PT-BR (ex.: *"Variáveis de ambiente obrigatórias ausentes:
  ..."*) exibida com `st.error`. Nenhum traceback cru aparece.

## Nota sobre o fluxo end-to-end

A estrutura, os imports e os caminhos de erro funcionam de forma isolada. Mas o
**fluxo completo de tool-use** (o Claude de fato consultando dados) exige **dois
pré-requisitos reais**:

1. um **backend VetFlow no ar** (local ou remoto) acessível em `VETFLOW_BASE_URL`
   com credenciais válidas; e
2. uma **`ANTHROPIC_API_KEY` real** com crédito.

Sem eles, o app sobe, navega e mostra os erros tratados -- mas não executa uma
conversa real com ferramentas.
