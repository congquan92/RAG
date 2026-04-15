import os
from dotenv import load_dotenv
def main() -> None:
    load_dotenv()
    key = os.environ.get("GOOGLE_AI_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_AI_API_KEY in environment")
    printed = False
    try:
        from google import genai as genai_new

        client = genai_new.Client(api_key=key)
        for model in client.models.list(config={"query_base": True}):
            actions = model.supported_actions or []
            if "embedContent" in actions or "batchEmbedContents" in actions:
                print(f"Model name: {model.name}")
                printed = True
    except ModuleNotFoundError:
        pass

    # Backward-compatible fallback for old SDK.
    if not printed:
        try:
            import google.generativeai as genai_old
            
            genai_old.configure(api_key=key)
            for model in genai_old.list_models():
                methods = model.supported_generation_methods or []
                if "embedContent" in methods:
                    print(f"Model name: {model.name}")
                    printed = True
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Neither google-genai nor google-generativeai is installed"
            ) from exc

    if not printed:
        print("No embedding-capable Gemini model found for this API key/project.")


if __name__ == "__main__":
    main()