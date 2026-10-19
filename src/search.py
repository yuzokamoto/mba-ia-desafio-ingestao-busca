"""Busca semântica + resposta da LLM restrita ao conteúdo do PDF.

Fluxo para cada pergunta: vetoriza a pergunta, busca os 10 chunks mais relevantes
no PGVector (`similarity_search_with_score`, k=10), monta o prompt com esses chunks
como CONTEXTO e chama a LLM.
"""

import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from providers import (
    SEARCH_K,
    ConfigError,
    count_chunks,
    get_llm,
    get_vector_store,
    load_settings,
)

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_documents(store, question: str, k: int = SEARCH_K):
    """Os k chunks mais próximos da pergunta, como pares (Document, score).

    O score é a distância de cosseno do PGVector: quanto menor, mais relevante.
    """
    return store.similarity_search_with_score(question, k=k)


def build_context(results) -> str:
    """Concatena o texto dos chunks recuperados, na ordem de relevância."""
    return "\n\n".join(doc.page_content.strip() for doc, _score in results)


def clean_answer(text: str) -> str:
    """Remove espaços e aspas que envolvem a resposta inteira (o prompt mostra respostas entre aspas)."""
    answer = text.strip()
    closing = {'"': '"', "'": "'", "“": "”"}
    if len(answer) >= 2 and closing.get(answer[0]) == answer[-1]:
        answer = answer[1:-1].strip()
    return answer


def search_prompt(question=None):
    """Monta a chain de busca + resposta.

    Sem `question`, devolve a chain (use `chain.invoke("pergunta")`); com `question`,
    já devolve a resposta em texto. Devolve None se a inicialização falhar.
    """
    try:
        settings = load_settings()
        total = count_chunks(settings)
        if total == 0:
            raise ConfigError(
                f"A collection '{settings.collection_name}' está vazia. "
                "Rode primeiro: python src/ingest.py"
            )
        store = get_vector_store(settings)
        llm = get_llm(settings)
    except Exception as exc:  # configuração, banco fora do ar, dependências...
        print(f"Erro ao iniciar a busca: {exc}", file=sys.stderr)
        return None

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    retrieve_context = RunnableLambda(lambda q: build_context(search_documents(store, q)))
    chain = (
        {"contexto": retrieve_context, "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(clean_answer)
    )

    if question is not None:
        return chain.invoke(question)
    return chain
