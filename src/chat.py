"""Chat no terminal: lê perguntas, consulta o PDF via busca semântica e imprime a resposta.

Uso (depois da ingestão):
    python src/chat.py
Digite "sair" (ou Ctrl+D / Ctrl+C) para encerrar.
"""

import sys

from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main() -> int:
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return 1

    print("Faça sua pergunta (digite 'sair' para encerrar):")
    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        try:
            answer = chain.invoke(question)
        except Exception as exc:  # erro pontual (rede, limite de requisições...) não encerra o chat
            print(f"RESPOSTA: [erro ao consultar o modelo: {exc}]")
            continue

        print(f"RESPOSTA: {answer}")
        print("\n---")

    print("Até mais!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
