import os
import weaviate
import uvicorn
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from graph import create_graph

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = "YOUR URL"
PDF_PATH = "ders2.pdf"
WEAVIATE_INDEX_NAME = "Dokumanlar"
MAX_ATTEMPTS = 1

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL, auth_credentials=AuthApiKey(api_key=WEAVIATE_API_KEY),
        headers={"X-Gemini-Api-Key": GEMINI_API_KEY}
    )
    print("Weaviate Cloud'a başarıyla bağlandı.")
except Exception as e:
    print(f"Weaviate'e bağlanırken hata oluştu: {e}")
    exit()

if not client.collections.exists(WEAVIATE_INDEX_NAME):
    print(f"'{WEAVIATE_INDEX_NAME}' indeksi Weaviate'te bulunamadı. PDF işleniyor...")
    if not os.path.exists(PDF_PATH):
        print(f"HATA: PDF dosyası bulunamadı: {PDF_PATH}")
        client.close()
        exit()
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(docs)
    WeaviateVectorStore.from_documents(
        client=client, documents=splitted_docs, embedding=embedding_model,
        index_name=WEAVIATE_INDEX_NAME, by_text=False
    )
    print("PDF başarıyla işlendi ve Weaviate'e yüklendi.")
else:
    print(f"'{WEAVIATE_INDEX_NAME}' indeksi zaten mevcut. Veri yükleme adımı atlandı.")

try:
    vectorstore = WeaviateVectorStore(
        client=client, index_name=WEAVIATE_INDEX_NAME, text_key='text', embedding=embedding_model
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    print("MultiQueryRetriever başarıyla oluşturuldu.")
except Exception as e:
    print(f"Retriever oluşturulurken hata: {e}")
    exit()

langgraph_app = create_graph(retriever=retriever, llm=llm, max_attempts=MAX_ATTEMPTS)

class AgentInput(BaseModel):
    question: str

class AgentOutput(BaseModel):
    generation: str

app = FastAPI(
    title="LangGraph RAG Agent Server",
    description="Döngüler ve karar mekanizmaları ile çalışan akıllı bir RAG ajanı"
)

@app.post("/agent", response_model=AgentOutput)
async def run_agent(input_data: AgentInput):
    graph_input = {"question": input_data.question}
    output = langgraph_app.invoke(graph_input)
    return output

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Ajanı</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f4f7f6; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.12); width: 100%; max-width: 600px; }
            h1 { text-align: center; color: #333; }
            textarea { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #ccc; font-size: 16px; margin-bottom: 20px; box-sizing: border-box; resize: vertical; min-height: 60px; }
            button { width: 100%; padding: 12px; border: none; background-color: #007bff; color: white; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; transition: background-color 0.2s; }
            button:hover { background-color: #0056b3; }
            #answer { margin-top: 20px; padding: 15px; background-color: #e9ecef; border-radius: 8px; border: 1px solid #dee2e6; min-height: 50px; white-space: pre-wrap; word-wrap: break-word; }
            #loading { text-align: center; display: none; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Akıllı RAG Ajanına Soru Sor</h1>
            <form id="qa-form">
                <textarea id="question" rows="3" placeholder="PDF ile ilgili sorunuzu buraya yazın..."></textarea>
                <button type="submit">Cevap Al</button>
            </form>
            <div id="loading">Cevap aranıyor, lütfen bekleyin...</div>
            <div id="answer"></div>
        </div>
        <script>
            document.getElementById('qa-form').addEventListener('submit', async function(event) {
                event.preventDefault();
                const question = document.getElementById('question').value;
                const answerDiv = document.getElementById('answer');
                const loadingDiv = document.getElementById('loading');

                if (!question) {
                    answerDiv.innerText = "Lütfen bir soru girin.";
                    return;
                }

                answerDiv.innerText = '';
                loadingDiv.style.display = 'block';

                try {
                    const response = await fetch('/agent', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question: question })
                    });

                    if (!response.ok) {
                        throw new Error(`Sunucu hatası: ${response.statusText}`);
                    }

                    const data = await response.json();
                    answerDiv.innerText = data.generation;

                } catch (error) {
                    answerDiv.innerText = 'Bir hata oluştu: ' + error.message;
                } finally {
                    loadingDiv.style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    try:
        print("\nAPI Sunucusu http://localhost:8000 adresinde başlatılıyor...")
        print("Ajanı kullanmak için tarayıcınızda bu adresi ziyaret edin.")
        uvicorn.run(app, host="localhost", port=8000)
    finally:
        print("\nSunucu kapatılıyor, Weaviate istemci bağlantısı kesiliyor.")
        client.close()
