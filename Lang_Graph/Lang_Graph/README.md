## LangGraph AI Travel Agent

This project is a Streamlit application. Run it locally with:

```powershell
streamlit run app.py
```

For Streamlit Cloud, set these values in **App settings > Secrets**:

```toml
OPENAI_API_KEY = "your-openai-key"
TAVILY_API_KEY = "your-tavily-key"
DUFFEL_ACCESS_TOKEN = "your-duffel-token"
```

`TAVILY_API_KEY` enables web search. `DUFFEL_ACCESS_TOKEN` enables flight and hotel searches. The app can still start with only `OPENAI_API_KEY` configured.
