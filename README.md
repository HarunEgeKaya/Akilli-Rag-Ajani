# Akilli-Rag-Ajani

#Bu projede gemini kullanılmaktadır. #Python sanal makina kurmanız gerekmektedir. python -m venv .venv

Bu sistem, pdf'deki metinleri vektör embedding'e çevirip Weaviate veritabanına kaydeder. FastAPI üzerinden gelen sorgular da embed edilir, en ilgili belgeler veritabanından alınır. LangChain ve LangGraph ile bu içerikler LLM'e aktarılır ve bağlama dayalı akıllı cevaplar üretilir.

.env
GEMINI_API_KEY=

LANGCHAIN_API_KEY=

LANGCHAIN_TRACING_V2=true

LANGCHAIN_PROJECT=SIMPLELLM

WEAVIATE_API_KEY=

WEAVIATE_INDEX_NAME=

WEAVIATE_URL = in main.py
