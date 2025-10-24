# main.py
import os
import textwrap
import sys
import google.generativeai as genai


def to_markdown(text: str) -> str:
    """Simple formatter: replace bullets and indent for console display."""
    text = text.replace("•", "  *")
    return textwrap.indent(text, "> ", predicate=lambda _: True)

def configure_genai():
    """Read API key from environment and configure the SDK."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Set it in your environment or use a .env file."
        )
    genai.configure(api_key=api_key)

def choose_model(preferred: str = "gemini-flash-latest") -> str:
    """Return a supported model name. If preferred isn't available, list models."""
    try:
        available = [m.name for m in genai.list_models()]
    except Exception as e:
        raise RuntimeError(f"Failed to list models: {e}")

    if preferred in available:
        return preferred

    # Try some common fallbacks if present
    for candidate in ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"):
        if candidate in available:
            return candidate

    # If nothing matched, raise with available models shown
    raise RuntimeError(
        f"Preferred model '{preferred}' not available. Models you can access:\n"
        + "\n".join(available)
    )

def run_chat_loop(model_name: str):
    """Start the chat loop with the configured model."""
    model = genai.GenerativeModel(
        model_name,
        system_instruction=(
            "You are an expert AI assistant specializing in Indian law. "
            "Explain complex legal topics in a clear, simple, and detailed manner "
            "for a general audience."
        ),
    )

    print("⚖️  Legal AI Assistant ⚖️")
    print("Ask me any question about Indian law. Type 'exit' or 'quit' to end.")
    print("-" * 40)

    chat = model.start_chat(history=[])

    while True:
        try:
            user_question = input("Your Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not user_question:
            continue

        if user_question.lower() in {"quit", "exit"}:
            print("Goodbye! 👋")
            break

        try:
            response = chat.send_message(user_question)
            print("\n--- Answer ---")
            print(to_markdown(response.text))
            print("---------------\n")
        except Exception as e:
            print(f"An error occurred when sending the message: {e}", file=sys.stderr)

def main():
    try:
        configure_genai()
    except RuntimeError as ex:
        print(f"Configuration error: {ex}", file=sys.stderr)
        sys.exit(1)

    try:
        model_name = choose_model(preferred="gemini-flash-latest")
    except RuntimeError as ex:
        print(f"Model selection error: {ex}", file=sys.stderr)
        sys.exit(1)

    try:
        run_chat_loop(model_name)
    except Exception as ex:
        print(f"Unexpected error: {ex}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
