# 🐾 vetflow-agent

[![CI](https://github.com/IannCarvalho/vetflow-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/IannCarvalho/vetflow-agent/actions/workflows/ci.yml)

Agente de IA **multi-provider** que responde perguntas em linguagem natural
sobre uma clínica veterinária, usando as ferramentas do servidor MCP
[`vetflow-mcp`](../vetflow-mcp) para consultar **dados reais** antes de
responder ("quais vacinas vencem esse mês?", "quantos pets a Maria tem?").

Expõe **duas interfaces**:

- **API HTTP** (`api.py`, FastAPI) — consumida pelo widget de chat do
  [`vetflow-frontend`](../vetflow-frontend). É o caminho "produto".
- **UI Streamlit** (`app.py`) — um app de chat autocontido, útil para testes
  rápidos sem o frontend.

O modelo (Claude, GPT, Gemini ou Bedrock) é escolhido por configuração — ver
*Multi-provider* abaixo.

> Este projeto **não modifica** o `vetflow-mcp`. Apenas o consome.

## Como conecta (arquitetura)

```mermaid
flowchart TD
    fe["Widget (React)"]
    st["app.py (Streamlit)"]
    api["api.py (FastAPI)<br/>/chat · /providers/models"]
    agent["agent.py<br/>loop de tool-use (OpenAI) · MAX_TURNS=12 · BYOK"]
    llm{{"LiteLLM → Claude / GPT / Gemini / Bedrock"}}
    bridge["mcp_bridge.py"]
    mcp["vetflow-mcp<br/>(python -m vetflow_mcp.server)"]
    be["Backend VetFlow (Go)"]
    db[("MongoDB")]

    fe -- "POST /chat" --> api
    st -- "run_agent()" --> agent
    api --> agent
    agent -- "litellm.acompletion" --> llm
    agent -- "dispatch_tool / list_tools" --> bridge
    bridge -- "stdio (subprocesso)" --> mcp
    mcp -- "Bearer JWT" --> be
    be --> db
```

- **`api.py`** (FastAPI) — `GET /health`, `GET /providers`,
  `POST /providers/models` (lista modelos reais de uma key, custo zero) e
  `POST /chat`. CORS configurável por `CORS_ORIGINS`.
- **`agent.py`** — loop de tool-use no formato OpenAI via `litellm.acompletion`:
  pede resposta ao modelo, executa cada `tool_call` (via a ponte MCP), devolve
  `role:"tool"` casado por `tool_call_id`, repete até parar (`MAX_TURNS = 12`).
  Aceita credencial **BYOK** por request (`model`/`api_key`), usada só na chamada
  e **nunca persistida**.
- **`provider.py`** — decide **qual LLM**: honra `LLM_MODEL`, senão auto-detecta
  pela credencial presente (Anthropic → OpenAI → Gemini → **Bedrock `.aws`**).
- **`mcp_bridge.py`** — sobe o `vetflow-mcp` como **subprocesso stdio**, descobre
  as tools e as converte para function-calling da OpenAI (que o LiteLLM usa).
- **`app.py`** — UI Streamlit alternativa (chat + sidebar sem segredos).

## Multi-provider (LiteLLM)

O agente roda sobre **qualquer LLM** suportado pelo LiteLLM:

- **Explícito:** `LLM_MODEL` no `.env` — ex. `gpt-4o`,
  `claude-3-5-sonnet-latest`, `gemini/gemini-2.0-flash`,
  `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0`.
- **Automático:** sem `LLM_MODEL`, usa o primeiro provider com credencial, nesta
  ordem: `ANTHROPIC_API_KEY` → `OPENAI_API_KEY` → `GEMINI_API_KEY` →
  **Amazon Bedrock** (perfil `.aws` default, sem declarar nada).
- **Por request (BYOK):** o frontend pode mandar `model`/`api_key` no `POST /chat`
  (chave do usuário, guardada só no navegador dele). Sobrescreve o padrão do
  servidor só naquela chamada.

> **Bedrock** não usa API key e sim credenciais AWS (`.aws`), por isso é um
> provider "do servidor", não BYOK. Ids de modelo on-demand antigos foram
> aposentados — use um **inference profile** (`bedrock/us.anthropic.claude-...`).

## Validação de key + listagem de modelos

`POST /providers/models {provider, api_key}` chama a API de listagem do provider
(**custo zero**, não gera tokens) e devolve os modelos **reais** que a key pode
usar, já filtrados para só os de chat. Lista vazia = key inválida. O widget usa
isso para validar a chave e popular o dropdown de modelos.

## Abordagem de integração escolhida

O `vetflow-mcp` é instalado em **modo editável no mesmo venv**:

```bash
pip install -e ../vetflow-mcp
```

Assim o subprocesso (`python -m vetflow_mcp.server`) roda com o **mesmo
interpretador** do app (`sys.executable`) e encontra o pacote `vetflow_mcp`.
Nada é duplicado: você edita o MCP no seu próprio repositório e o agente passa a
enxergar as mudanças na próxima execução.

### Por que o `vetflow-mcp` é um projeto separado (e não código deste repo)

O agente é o principal consumidor do MCP, então é natural perguntar se o MCP não
deveria morar *dentro* dele. Não deveria: o `vetflow-mcp` é um **servidor** MCP e
este agente é apenas **um cliente** entre vários — Claude Desktop e outros
clientes MCP usam aquele servidor **sem este agente existir**. Fundir os dois
mataria a reutilização que é a razão de ser do MCP (cada cliente teria que
reimplementar o acoplamento com o backend).

Por isso o agente **consome** o MCP sem fundir código: em dev via `pip install -e
../vetflow-mcp`, no deploy via git **submodule** (`vendor/vetflow-mcp`, build
autocontido). O custo de ter dois *processos* em runtime (o subprocesso stdio)
não se paga fundindo repositórios e sim movendo o MCP para **in-process** quando a
escala exigir — ver *"Como o MCP é consumido: subprocesso vs. in-process"* abaixo.

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
# edite o .env: uma chave de provider (ANTHROPIC/OPENAI/GEMINI) OU perfil .aws
# para Bedrock, + VETFLOW_BASE_URL/EMAIL/PASSWORD
```

Rode a **API** (para o frontend consumir):

```bash
uvicorn api:app --host 127.0.0.1 --port 8700
```

Ou a **UI Streamlit** (teste rápido sem frontend):

```bash
streamlit run app.py
```

## Variáveis de ambiente

| Variável | Papel |
|---|---|
| `LLM_MODEL` | Força um modelo/provider (ex. `gpt-4o`, `gemini/gemini-2.0-flash`). Opcional — sem ela, auto-detecta. |
| `ANTHROPIC_API_KEY` | Provider Anthropic (Claude). |
| `OPENAI_API_KEY` | Provider OpenAI (GPT). |
| `GEMINI_API_KEY` | Provider Google (Gemini). |
| `AWS_PROFILE` / `AWS_REGION` / `BEDROCK_MODEL_ID` | Bedrock (opcional; usa o perfil `.aws` default). |
| `CORS_ORIGINS` | Origens permitidas na API (default cobre o dev do VetFlow/Vite). |
| `VETFLOW_BASE_URL` | URL do backend VetFlow (ex.: `http://localhost:8080`). |
| `VETFLOW_EMAIL` / `VETFLOW_PASSWORD` | Login do vet; o MCP autentica no backend. |
| `VETFLOW_TIMEOUT_S` | Timeout HTTP em segundos (default 15). |
| `VETFLOW_ENABLE_WRITES` | Liga as tools de escrita do MCP (default `false`). |

As credenciais de LLM podem vir do ambiente (acima) **ou** por request (BYOK, o
widget manda a key do usuário). Todas são carregadas no topo do `app.py`/`api.py`
antes de importar o agente, para chegarem também ao subprocesso MCP.

## Deploy (Render)

Há um blueprint em [`render.yaml`](./render.yaml) para publicar a API no
[Render](https://render.com). No Render: **New → Blueprint**, aponte para este
repositório, e preencha no painel as variáveis `sync:false`.

Pontos do deploy:

- **BYOK-only:** o servidor **não** carrega nenhuma chave de LLM. Quem usar o
  chat traz a própria key (enviada por request, guardada no navegador). Assim o
  deploy não tem custo de LLM nem guarda credenciais de terceiros.
- **`vetflow-mcp` via git submodule:** o MCP vem em `vendor/vetflow-mcp` como
  **submodule** (continua sendo um repositório separado). O build roda
  `pip install ./vendor/vetflow-mcp`, sem clone externo anônimo. O Render clona
  submodules com a credencial da sua integração GitHub, então **funciona mesmo
  com o `vetflow-mcp` privado**. Ao clonar para dev local, use
  `git clone --recursive` (ou `git submodule update --init`).
- **Variáveis a preencher:** apenas `CORS_ORIGINS` (a URL do frontend no Vercel).
  `VETFLOW_BASE_URL` já vem no blueprint. **Não há `VETFLOW_EMAIL/PASSWORD`**: a
  auth é pass-through (o JWT do usuário logado chega pelo header do `/chat`).
- **Frontend:** defina `VITE_AGENT_API_URL` (no Vercel) com a URL pública da API
  do agente no Render, para o widget encontrá-la em produção.

> Sem este deploy, o widget em produção mostra *"Não consegui falar com o
> assistente (Network Error)"* — o front está no ar, mas a API do agente não.

## Como o MCP é consumido: subprocesso vs. in-process

Esta é a decisão de arquitetura mais importante do agente e vale entender o
porquê, porque ela define o **perfil de memória** em produção.

Hoje, cada `POST /chat` sobe o `vetflow-mcp` como **subprocesso stdio próprio**
(`python -m vetflow_mcp.server`), fala com ele pelo `mcp_bridge.py`, e o fecha ao
terminar. Ou seja: durante uma conversa com ferramentas, **dois interpretadores
Python coexistem** — o do agente (FastAPI + LiteLLM) e o do MCP.

### As duas abordagens

| | **Subprocesso por request** (atual) | **In-process** (alternativa) |
|---|---|---|
| Como roda | o MCP é um processo separado, falado por stdio | o agente importa `vetflow_mcp` e chama `list_tools()`/`call_tool()` direto |
| Isolamento | **total** — cada request tem seu processo e seu token; um crash do MCP não derruba a API | compartilhado — tudo roda no mesmo processo |
| Multi-tenancy | **seguro por construção** — processos não compartilham estado, então o JWT de um usuário nunca vaza para outro | **exige cuidado** — o `server.py` do MCP usa um `_client` singleton global; in-process ingênuo reabriria o furo de multi-tenancy |
| Memória | **pico alto** — dois Python residentes ao mesmo tempo (o segundo custa ~80MB) | **enxuto** — um só interpretador |
| Latência | custo de subir o subprocesso a cada request | sem spawn; chamada de função direta |

### Por que o subprocesso foi escolhido

**Isolamento e segurança de multi-tenancy, de graça.** Como cada request tem seu
próprio processo, o token (JWT) de um usuário fisicamente não pode vazar para a
request de outro — não há estado compartilhado para vazar. Para um serviço que
age com a identidade de quem está logado (ver *pass-through* na seção de deploy),
essa garantia vale muito e sai sem escrever uma linha de código de isolamento.

### O custo, e quando ele morde

O preço é a **RAM de pico**: no plano free do Render (**512MB**), os dois
interpretadores Python coexistindo durante uma chamada de ferramenta chegam
perto do teto. Isso já causou **OOM-kill** (`502` com header
`x-render-routing: no-deploy`, logs subindo → `Shutting down` sem traceback →
reiniciar). Duas mudanças aliviaram o pico sem mexer na arquitetura:

- **fail-fast de provider:** `run_agent` resolve o provider *antes* de importar
  o LiteLLM e abrir o MCP — uma request sem provider não carrega os dois pesos à
  toa;
- **runtime enxuto:** o `streamlit` saiu do `requirements.txt` de deploy (ele
  arrasta pandas/pyarrow/numpy/altair, que a API nunca importa) e foi para o
  `requirements-dev.txt`.

Com esses dois cortes o serviço roda **estável no free tier** e já respondeu um
`POST /chat` real end-to-end (front → agente → MCP → backend). O subprocesso
segue sendo a abordagem em produção.

### O caminho de saída, se precisar escalar

Se a carga crescer e a RAM de pico voltar a ser o gargalo, a cura real é mover o
MCP para **in-process** — eliminando o segundo interpretador. O bloqueio é o
`_client` singleton global do `server.py`; a solução acordada (**Opção A**) é usar
`contextvars` para manter o client/token **por request**, preservando o
isolamento de multi-tenancy sem subir um processo. É um PR no `vetflow-mcp` + uma
troca no `mcp_bridge.py`. O modo subprocesso **permanece** de qualquer forma —
ele é como o `vetflow-mcp` serve o Claude Desktop e outros clientes stdio.

> **Resumo:** subprocesso = mais RAM, isolamento grátis, em produção hoje.
> In-process = enxuto, exige `contextvars` para não reabrir o furo de
> multi-tenancy, reservado para quando a escala exigir.

## Degradação graciosa (sem backend / sem chave)

O app foi feito para **não quebrar** quando falta configuração:

- **Sem nenhum provider** (nenhuma chave e sem `.aws`): a entrada de chat é
  desabilitada com um aviso; a UI continua navegável.
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
2. uma **credencial de LLM** real com crédito — uma chave de provider
   (`ANTHROPIC`/`OPENAI`/`GEMINI`) ou um perfil AWS com acesso ao Bedrock.

Sem eles, o app sobe, navega e mostra os erros tratados -- mas não executa uma
conversa real com ferramentas.
