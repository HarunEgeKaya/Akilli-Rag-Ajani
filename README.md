# Akilli-Rag-Ajani
Bu sistem, metinleri vektör embedding'e çevirip Weaviate veritabanına kaydeder. FastAPI üzerinden gelen sorgular da embed edilir, en ilgili belgeler veritabanından alınır. LangChain ve LangGraph ile bu içerikler LLM'e aktarılır ve bağlama dayalı akıllı cevaplar üretilir.

.env
GEMINI_API_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=SIMPLELLM
WEAVIATE_API_KEY=
WEAVIATE_INDEX_NAME=

WEAVIATE_URL = in main.py
