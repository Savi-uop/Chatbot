import os
from dotenv import load_dotenv

# 1. Load API Key from .env
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def run_rag():
    # 2. Load and Split PDF
    # loader = PyPDFLoader("data/AI_Strategy_2026.pdf")
    loader = PyPDFLoader("data/Global Food Systems.pdf")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 3. Create Embeddings (Using current stable model)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 4. Initialize Vector Store
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # 5. Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 6. Define Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
    """)
    
    # 7. Create Retrieval Chain (The modern LCEL way)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(), 
        combine_docs_chain
    )

    # 8. Start the Chat Loop
    print("\n--- AI Chatbot Ready! (Type 'exit' to stop) ---")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Invoke the chain with the user input
        result = retrieval_chain.invoke({"input": user_input})
        
        print(f"AI: {result['answer']}")

if __name__ == "__main__":
    run_rag()