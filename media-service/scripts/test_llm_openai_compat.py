import asyncio
import sys

# Allow running this script from repo root or media-service/
sys.path.append("media-service")

# Importing media_service triggers .env auto-loading (repo root or media-service/.env)
import src.media_service  # noqa: F401

from src.media_service import get_system_prompt
from src.ai_providers import create_llm_client


async def main() -> None:
    system_prompt = get_system_prompt()
    client = create_llm_client("openai_compat", system_prompt=system_prompt)

    history: list[dict] = []
    first_line = system_prompt.splitlines()[0] if system_prompt else ""
    print(f"OpenAI-compatible chat test. System prompt loaded: {first_line!r}")
    print("Type 'exit' to quit.")

    while True:
        user_text = (await asyncio.to_thread(input, "You> ")).strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", ":q"}:
            break

        try:
            # Pass the history; the client will append the user message if needed.
            reply = await client.generate(user_text, history)
        except Exception as e:
            print(f"[error] {e}")
            continue

        print(f"AI> {reply}")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    asyncio.run(main())

