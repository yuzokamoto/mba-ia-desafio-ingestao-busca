# Ingestão e Busca Semântica com LangChain e Postgres

Desafio técnico do MBA em Engenharia de Software com IA (Full Cycle).

O software tem duas partes:

- **Ingestão** (`src/ingest.py`): lê o `document.pdf`, divide o texto em chunks de 1000 caracteres com
  overlap de 150, gera um embedding por chunk e grava os vetores no PostgreSQL com pgVector.
- **Busca via CLI** (`src/chat.py`): para cada pergunta, vetoriza a pergunta, busca os 10 chunks mais
  relevantes (`similarity_search_with_score(query, k=10)`), monta o prompt do enunciado com esses chunks
  como CONTEXTO e chama a LLM. A resposta vem apenas do conteúdo do PDF; fora dele, a resposta é
  "Não tenho informações necessárias para responder sua pergunta."

Enunciado original: [devfullcycle/mba-ia-desafio-ingestao-busca](https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca).

## Exemplo de uso

```
$ python src/chat.py
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: R$ 10.000.000,00

---

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

---

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

---

PERGUNTA: sair
Até mais!
```

## Pré-requisitos

- Python 3.10 ou superior (testado com 3.14)
- Docker e Docker Compose
- Uma API key da **OpenAI** ou do **Google (Gemini)**

## Como executar

Todos os comandos são executados a partir da raiz do projeto.

### 1. Ambiente virtual e dependências

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e preencha **uma** das chaves:

- OpenAI: `OPENAI_API_KEY=...`
- Gemini: `GOOGLE_API_KEY=...`

As demais variáveis já vêm com valores que funcionam com o `docker-compose.yml`:

| Variável | Padrão | Descrição |
|---|---|---|
| `LLM_PROVIDER` | vazio | `openai` ou `gemini`. Vazio = usa o provedor cuja chave estiver preenchida (OpenAI tem prioridade). |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings da OpenAI |
| `OPENAI_LLM_MODEL` | `gpt-6-luna` | Modelo que responde às perguntas (OpenAI) |
| `GOOGLE_EMBEDDING_MODEL` | `gemini-embedding-001` | Modelo de embeddings do Gemini |
| `GOOGLE_LLM_MODEL` | `gemini-3.5-flash-lite` | Modelo que responde às perguntas (Gemini) |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` | Conexão com o Postgres do Docker |
| `PG_VECTOR_COLLECTION_NAME` | `document_chunks` | Nome da collection no PGVector |
| `PDF_PATH` | `document.pdf` | PDF a ser ingerido (relativo à raiz do projeto) |

### 3. Subir o banco de dados

```bash
docker compose up -d
```

Sobe o PostgreSQL 17 com pgVector na porta 5432; o serviço `bootstrap_vector_ext` cria a extensão `vector`.

### 4. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada:

```
Ingerindo document.pdf (provedor: openai, embeddings: text-embedding-3-small)...
Ingestão concluída: 67 chunks gravados na collection 'document_chunks'.
```

### 5. Rodar o chat

```bash
python src/chat.py
```

Digite `sair` (ou Ctrl+D) para encerrar.

## Como funciona

```
document.pdf ──PyPDFLoader──> páginas ──RecursiveCharacterTextSplitter(1000, 150)──> 67 chunks
      ──embeddings──> PGVector (PostgreSQL + pgVector, collection document_chunks)

pergunta ──embedding──> similarity_search_with_score(k=10) ──> 10 chunks concatenados = CONTEXTO
      ──> PromptTemplate (prompt do enunciado) ──> LLM ──> resposta
```

| Arquivo | Responsabilidade |
|---|---|
| `src/ingest.py` | Carrega o PDF (`PyPDFLoader`), divide em chunks (`RecursiveCharacterTextSplitter`, 1000/150), gera os embeddings e grava no PGVector. |
| `src/search.py` | Contém o `PROMPT_TEMPLATE` do enunciado e `search_prompt()`, que monta a chain: busca k=10 → contexto → prompt → LLM → texto. |
| `src/chat.py` | Loop de perguntas e respostas no terminal. |
| `src/providers.py` | Lê o `.env`, escolhe o provedor e cria embeddings, LLM e vector store. É compartilhado para que ingestão e busca usem sempre o mesmo modelo de embeddings e a mesma collection. |

Detalhes de implementação:

- **Reingestão segura**: a cada `python src/ingest.py` a collection é apagada e recriada, então rodar de
  novo não duplica chunks. Os embeddings são gerados antes de mexer no banco: se a API falhar (chave
  inválida, limite de requisições), os dados já ingeridos continuam intactos.
- **IDs determinísticos** (`document-0000`, `document-0001`, …) e metadados da página de origem em cada chunk.
- **Erros claros na inicialização do chat**: falta de chave, banco fora do ar ou collection vazia
  ("Rode primeiro: python src/ingest.py") são informados antes da primeira pergunta.

## Escolha dos modelos

Modelos consultados na documentação oficial em outubro de 2026, priorizando os mais leves e baratos de cada provedor.

**Embeddings**

- OpenAI: `text-embedding-3-small` (1536 dimensões, US$ 0,02 por 1M tokens), o modelo de embeddings
  econômico atual da OpenAI.
- Gemini: `gemini-embedding-001` (3072 dimensões), modelo estável de embeddings de texto. O
  `models/embedding-001` que vinha no `.env.example` original foi desligado pelo Google em outubro de 2025.

**LLM (OpenAI)**: comparei três modelos baratos com o prompt e a busca do desafio, em 7 perguntas (4 com
resposta no PDF e 3 fora de contexto). Os três acertaram todas:

| Modelo | Preço (entrada / saída por 1M tokens) | Tempo médio por pergunta | Observação |
|---|---|---|---|
| `gpt-6-luna` (padrão) | US$ 0,10 / US$ 0,50 | ~1,4 s | Respostas diretas e exatas |
| `gpt-5.4-nano` | US$ 0,20 / US$ 1,25 | ~1,3 s | Responde com negrito em Markdown, que aparece como `**` no terminal |
| `gpt-5-nano` | US$ 0,05 / US$ 0,40 | ~4,0 s | Gasta tokens de raciocínio (~3x mais lento) e às vezes acrescenta dados de outra coluna (ex.: o ano de fundação junto do faturamento) |

O `gpt-6-luna` é o modelo que a OpenAI indica para tarefas focadas e de alto volume com custo baixo.
Para trocar de modelo, altere `OPENAI_LLM_MODEL` no `.env`.

**LLM (Gemini)**: `gemini-3.5-flash-lite`, o Flash-Lite mais recente indicado pelo Google para novos
projetos. Esse modelo não aceita temperatura customizada, por isso a temperatura fica no padrão do provedor.

O projeto foi testado de ponta a ponta com OpenAI. O Gemini é suportado com `LLM_PROVIDER=gemini` e `GOOGLE_API_KEY`.

## Trocar o modelo de embeddings

Modelos de embeddings diferentes geram vetores de dimensões diferentes (1536 na OpenAI, 3072 no Gemini).
Como a ingestão apaga e recria a collection, basta rodar `python src/ingest.py` de novo depois de trocar o
modelo ou o provedor no `.env`. A busca precisa usar o mesmo modelo da ingestão.

Para zerar o banco por completo:

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```

## Problemas comuns

| Mensagem | Causa / solução |
|---|---|
| `Nenhuma chave de API encontrada` | Preencha `OPENAI_API_KEY` ou `GOOGLE_API_KEY` no `.env`. |
| `connection failed ... Connection refused` | O banco não está no ar: `docker compose up -d`. |
| `A collection 'document_chunks' está vazia` | Rode `python src/ingest.py` antes do chat. |
| `permission denied ... docker.sock` | Seu usuário não está no grupo `docker`: `sudo usermod -aG docker $USER` e abra uma nova sessão. |

## Estrutura do projeto

```
├── docker-compose.yml    # PostgreSQL 17 + pgVector
├── requirements.txt      # Dependências (versões fixadas, Python 3.10+)
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca (prompt + chain)
│   ├── chat.py           # CLI para interação com usuário
│   └── providers.py      # Configuração compartilhada (provedor, modelos, PGVector)
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução
```
