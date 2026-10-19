"""Configuração compartilhada entre ingestão e busca.

Centraliza a leitura do `.env`, a escolha do provedor (OpenAI ou Gemini) e a criação
dos objetos do LangChain (embeddings, LLM e vector store), para que `ingest.py` e
`search.py` usem exatamente o mesmo modelo de embeddings e a mesma collection.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# O .env fica na raiz do projeto; carregá-lo por caminho absoluto permite rodar os
# scripts de qualquer diretório.
load_dotenv(PROJECT_ROOT / ".env")

SUPPORTED_PROVIDERS = ("openai", "gemini")

# Modelos padrão (leves e baratos), usados quando o .env não define outro.
# Critérios da escolha e comparação entre modelos: ver README.
DEFAULT_MODELS = {
    "openai": {"embedding": "text-embedding-3-small", "llm": "gpt-6-luna"},
    "gemini": {"embedding": "gemini-embedding-001", "llm": "gemini-3.5-flash-lite"},
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_K = 10


class ConfigError(RuntimeError):
    """Configuração ausente ou inválida no .env."""


@dataclass(frozen=True)
class Settings:
    provider: str
    embedding_model: str
    llm_model: str
    database_url: str
    collection_name: str
    pdf_path: Path


def _env(name: str, default: str = "") -> str:
    """Lê uma variável de ambiente, tratando string vazia como ausente."""
    value = os.getenv(name, "").strip()
    return value or default


def _resolve_provider() -> str:
    provider = _env("LLM_PROVIDER").lower()
    if provider:
        if provider not in SUPPORTED_PROVIDERS:
            raise ConfigError(
                f"LLM_PROVIDER inválido: {provider!r}. Use um de: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        return provider
    # Sem LLM_PROVIDER explícito: usa o provedor cuja chave estiver preenchida.
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("GOOGLE_API_KEY"):
        return "gemini"
    raise ConfigError(
        "Nenhuma chave de API encontrada. Preencha OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env."
    )


def load_settings() -> Settings:
    """Monta as configurações a partir do .env e valida o que é obrigatório."""
    provider = _resolve_provider()
    defaults = DEFAULT_MODELS[provider]

    if provider == "openai":
        if not _env("OPENAI_API_KEY"):
            raise ConfigError("LLM_PROVIDER=openai, mas OPENAI_API_KEY está vazia no .env.")
        embedding_model = _env("OPENAI_EMBEDDING_MODEL", defaults["embedding"])
        llm_model = _env("OPENAI_LLM_MODEL", defaults["llm"])
    else:
        if not _env("GOOGLE_API_KEY"):
            raise ConfigError("LLM_PROVIDER=gemini, mas GOOGLE_API_KEY está vazia no .env.")
        embedding_model = _env("GOOGLE_EMBEDDING_MODEL", defaults["embedding"])
        llm_model = _env("GOOGLE_LLM_MODEL", defaults["llm"])

    database_url = _env("DATABASE_URL")
    if not database_url:
        raise ConfigError("DATABASE_URL não definida no .env.")
    collection_name = _env("PG_VECTOR_COLLECTION_NAME")
    if not collection_name:
        raise ConfigError("PG_VECTOR_COLLECTION_NAME não definida no .env.")

    pdf_path = Path(_env("PDF_PATH", "document.pdf"))
    if not pdf_path.is_absolute():
        # Caminho relativo é sempre relativo à raiz do projeto, não ao diretório atual.
        pdf_path = PROJECT_ROOT / pdf_path

    return Settings(
        provider=provider,
        embedding_model=embedding_model,
        llm_model=llm_model,
        database_url=database_url,
        collection_name=collection_name,
        pdf_path=pdf_path,
    )


def get_embeddings(settings: Settings):
    """Modelo de embeddings do provedor escolhido (o mesmo na ingestão e na busca)."""
    if settings.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model)


def get_llm(settings: Settings):
    """LLM que responde às perguntas a partir do contexto recuperado.

    A temperatura fica no padrão de cada modelo: os modelos de raciocínio da OpenAI e os
    Gemini mais novos (ex.: gemini-3.5-flash-lite) não aceitam temperatura customizada.
    """
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llm_model)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=settings.llm_model)


def get_vector_store(settings: Settings, embeddings=None, pre_delete_collection: bool = False):
    """Vector store PGVector apontando para a collection configurada.

    `pre_delete_collection=True` apaga a collection antes de usá-la; só a ingestão
    deve passar True (reingestão do zero, inclusive ao trocar o modelo de embeddings).
    """
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=embeddings if embeddings is not None else get_embeddings(settings),
        collection_name=settings.collection_name,
        connection=settings.database_url,
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )


def count_chunks(settings: Settings) -> int:
    """Quantos chunks a collection tem no banco (0 se ela ainda não existe)."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import ProgrammingError

    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as conn:
            return conn.execute(
                text(
                    "SELECT count(*) FROM langchain_pg_embedding e "
                    "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                    "WHERE c.name = :name"
                ),
                {"name": settings.collection_name},
            ).scalar_one()
    except ProgrammingError:
        # Tabelas do langchain_postgres ainda não criadas: nada foi ingerido.
        return 0
    finally:
        engine.dispose()
