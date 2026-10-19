"""Ingestão do PDF: carrega, divide em chunks, gera embeddings e grava no PGVector.

Uso (a partir da raiz do projeto, com o banco no ar):
    python src/ingest.py
"""

import logging
import sys
import warnings

# O enunciado recomenda o PyPDFLoader do langchain_community; o pacote avisa na importação
# que está em descontinuação. O aviso não afeta o funcionamento, então é silenciado aqui.
warnings.filterwarnings("ignore", message=r".*langchain-community.*", category=DeprecationWarning)

# Na primeira ingestão o PGVector loga "Collection not found" ao tentar apagar a collection
# que ainda não existe; é esperado, então só erros reais do langchain_postgres aparecem.
logging.getLogger("langchain_postgres").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from providers import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ConfigError,
    Settings,
    get_embeddings,
    get_vector_store,
    load_settings,
)


def load_and_split(pdf_path) -> list[Document]:
    """Lê o PDF e o divide em chunks de 1000 caracteres com overlap de 150."""
    pages = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)

    # Mantém só metadados com valor (o PyPDFLoader traz vários campos vazios do PDF).
    return [
        Document(
            page_content=chunk.page_content,
            metadata={k: v for k, v in chunk.metadata.items() if v not in ("", None)},
        )
        for chunk in chunks
        if chunk.page_content.strip()
    ]


def ingest_pdf(settings: Settings | None = None) -> int:
    """Executa a ingestão completa e devolve a quantidade de chunks gravados."""
    settings = settings or load_settings()
    if not settings.pdf_path.is_file():
        raise ConfigError(f"PDF não encontrado: {settings.pdf_path}")

    chunks = load_and_split(settings.pdf_path)
    if not chunks:
        raise ConfigError(f"Nenhum texto extraído de {settings.pdf_path}")

    # Gera os embeddings de todos os chunks antes de tocar no banco: se a API falhar
    # (chave inválida, limite de requisições), a collection existente fica intacta.
    embeddings = get_embeddings(settings)
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    # A collection é recriada a cada ingestão: rodar de novo não duplica chunks e
    # trocar o modelo de embeddings não mistura vetores de dimensões diferentes.
    store = get_vector_store(settings, embeddings=embeddings, pre_delete_collection=True)
    store.add_embeddings(
        texts=texts,
        embeddings=vectors,
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[f"{settings.pdf_path.stem}-{i:04d}" for i in range(len(chunks))],
    )
    return len(chunks)


def main() -> int:
    try:
        settings = load_settings()
        print(f"Ingerindo {settings.pdf_path.name} (provedor: {settings.provider}, "
              f"embeddings: {settings.embedding_model})...")
        total = ingest_pdf(settings)
    except ConfigError as exc:
        print(f"Erro de configuração: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # falhas de rede, banco fora do ar, chave inválida...
        print(f"Falha na ingestão: {exc}", file=sys.stderr)
        return 1

    print(f"Ingestão concluída: {total} chunks gravados na collection "
          f"'{settings.collection_name}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
