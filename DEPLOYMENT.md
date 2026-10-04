# Deployment Checklist

This project is fully designed and optimized to be deployed on **Streamlit Community Cloud** (or other similar containerized PaaS solutions). 

## Prerequisites
- A GitHub account
- A Streamlit Community Cloud account (sign up at [share.streamlit.io](https://share.streamlit.io/))
- Your OpenAI or OpenRouter API keys

## GitHub Setup
1. Ensure all your local changes are committed.
2. Ensure the `.gitignore` correctly ignores `.env`, `venv/`, and `__pycache__/` (this is already configured).
3. Push your repository to GitHub. Make sure the repository is either public or private, but your Streamlit account must have access to it.

## Streamlit Deployment Settings
1. Go to your Streamlit dashboard and click **New App**.
2. Select your repository, branch (usually `main`), and the entry point file.
   - **Main file path:** `app/ui.py`
3. Click **Advanced settings**.

## Required Secrets
Before deploying, you must configure your secrets. In the **Advanced settings** (or Settings > Secrets), add the following in TOML format:

```toml
OPENROUTER_API_KEY="your-openrouter-api-key"
# OR
OPENAI_API_KEY="your-openai-api-key"
```

*Note: The application is configured to securely read from `st.secrets` during deployment and fallback to `.env` during local development.*

## Deployment Audit Checklist Complete
- [x] **Entry Point:** The application safely boots via `streamlit run app/ui.py`.
- [x] **Requirements:** `requirements.txt` contains exactly what is needed (`streamlit`, `sentence-transformers`, `faiss-cpu`, `openai`, `pypdf`, `rank_bm25`, `python-dotenv`). No unnecessary local bloat.
- [x] **Environment Variables:** API keys are never hardcoded. Handled safely via `os.environ` and `st.secrets`.
- [x] **Caching/Performance:** Heavy models (`SentenceTransformer` and `CrossEncoder`) use `@st.cache_resource` for singleton behavior and fast app reloading.
- [x] **Temporary Storage:** `app/ui.py` utilizes `tempfile.TemporaryDirectory()`. The app makes no assumptions about persistent disk space, making it perfect for stateless containers.
- [x] **Error Handling:** Try/except blocks catch missing keys, empty PDFs, and parsing errors, showing user-friendly `st.error()` or `st.warning()` messages instead of stack traces.
- [x] **Session State:** Vector stores and processed documents are bound to `st.session_state` so they persist seamlessly across user interactions.

## Testing After Deployment
1. Wait for the app to "Bake" (Streamlit will install dependencies).
2. Once the app loads, verify you do not see the "API Key is missing!" warning.
3. Upload a sample PDF and click "Process Documents".
4. Ask a question. Ensure the LLM correctly generates an answer with citations. 
