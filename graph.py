from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# --- 1. Grafiğin Hafızasını (State) Tanımla ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    _decision: str
    attempts: int

# --- 2. Grafiğin Düğümlerini (Nodes) Tanımla ---

def retrieve(state: GraphState, retriever, llm) -> GraphState:
    print("---DÜĞÜM: RETRIEVE---")
    question = state["question"]
    attempts = state.get("attempts", 0)
    print(f"   -> Deneme #{attempts + 1}")
    documents = retriever.invoke(question)
    print(f"   -> {len(documents)} adet döküman bulundu.")
    return {"documents": documents, "question": question, "generation": "", "_decision": "", "attempts": attempts}

def grade_documents(state: GraphState, llm) -> GraphState:
    print("---DÜĞÜM: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    class Grade(BaseModel):
        binary_score: str = Field(description="Are the documents relevant to the question? Answer 'yes' or 'no'.")
    parser = JsonOutputParser(pydantic_object=Grade)
    prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "If the document contains keywords or concepts related to the user question, grade it as relevant.\n"
        "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.\n\n"
        "JSON Format Instructions: {format_instructions}\n\n"
        "Retrieved Document:\n{documents}\n\nUser Question: {question}"
    )
    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser
    docs_str = "\n\n".join(doc.page_content for doc in documents)
    try:
        grade_result = chain.invoke({"question": question, "documents": docs_str})
        decision = grade_result.get('binary_score', 'no').lower()
        print(f"   -> Değerlendirme sonucu: {decision}")
        state['_decision'] = decision
    except Exception as e:
        print(f"   -> Değerlendirme sırasında hata oluştu: {e}")
        state['_decision'] = "no"
    return state

def generate(state: GraphState, llm) -> GraphState:
    print("---DÜĞÜM: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    docs_str = "\n\n".join(doc.page_content for doc in documents)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert assistant that analyzes and summarizes technical documents. "
        "Use the following context to answer the user's question. "
        "Synthesize the information into a coherent paragraph or bullet points, highlighting only the most important points. "
        "IMPORTANT: Answer the user in the same language as their original question.\n\n"
        "CONTEXT:\n{context}\n\n"
        "USER QUESTION: {question}"
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    generation = chain.invoke({"context": docs_str, "question": question})
    print("   -> Cevap üretildi.")
    print(f"\n--- NİHAİ CEVAP ---\n{generation}\n-------------------\n")
    return {"documents": documents, "question": question, "generation": generation, "_decision": state['_decision'], "attempts": state['attempts']}

def transform_query(state: GraphState, llm) -> GraphState:
    print("---DÜĞÜM: TRANSFORM QUERY---")
    question = state["question"]
    attempts = state["attempts"]
    prompt = ChatPromptTemplate.from_template(
        "You are a query transformation expert. Your task is to rephrase a user's question into a more effective search query for a vector database. "
        "Focus on extracting keywords and concepts. For a generic question like 'What is this document about?', reformulate it to search for a table of contents or main topics. "
        "For example:\n"
        "Original: 'What are the main topics of the PDF?' -> New Query: 'Table of Contents, Index, Main sections, Abstract'\n"
        "Original: 'Tell me about init' -> New Query: 'SysV init process, runlevels, /etc/inittab'\n\n"
        "Now, transform the following question:\nOriginal Question: {question}"
    )
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})
    print(f"   -> Yeni soru: {better_question}")
    return {"documents": [], "question": better_question, "generation": "", "attempts": attempts + 1}

def handle_failure(state: GraphState) -> GraphState:
    print("---DÜĞÜM: HANDLE FAILURE---")
    generation = "Birden fazla denemeye rağmen sorunuza dökümanlarda net bir cevap bulamadım."
    print(f"\n--- NİHAİ CEVAP ---\n{generation}\n-------------------\n")
    return {"generation": generation}

# --- 3. Grafiğin Karar Mekanizmasını Tanımla ---

def decide_to_generate(state: GraphState, max_attempts: int) -> str:
    print("---KARAR NOKTASI---")
    if state['_decision'] == "yes":
        print("   -> Karar: Dökümanlar yeterli. Cevap üretilecek.")
        return "generate"
    else:
        if state["attempts"] >= max_attempts:
            print(f"   -> Karar: Tekrar deneme hakkı ({max_attempts}) bitti. Cevap üretilemiyor.")
            return "handle_failure"
        else:
            print("   -> Karar: Dökümanlar yetersiz. Soru yeniden formüle edilecek.")
            return "transform_query"



def create_graph(retriever, llm, max_attempts):
    from functools import partial

    retrieve_node = partial(retrieve, retriever=retriever, llm=llm)
    grade_documents_node = partial(grade_documents, llm=llm)
    generate_node = partial(generate, llm=llm)
    transform_query_node = partial(transform_query, llm=llm)
    decide_to_generate_node = partial(decide_to_generate, max_attempts=max_attempts)

    # Akış şemasını çiziyoruz
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("handle_failure", handle_failure)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", decide_to_generate_node,
        {"generate": "generate", "transform_query": "transform_query", "handle_failure": "handle_failure"}
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("handle_failure", END)

    return workflow.compile()
