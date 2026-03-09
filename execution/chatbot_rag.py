import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Configuration
KB_DIR = "knowledge_base"
DB_DIR = ".chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Strict Guardrail Prompt
SYSTEM_PROMPT = """You are an AI assistant for Charlyn Ooi's training programs (Accelerator Program, Momentum Club, and Leads on Autopilot). 

Your instructions are absolute:
1. You MUST answer the user's question using ONLY the provided context excerpts from the training material.
2. DO NOT guess, hallucinate, or rely on outside internet knowledge.
3. DO NOT invent strategies, features, or advice that are not explicitly stated in the context.
4. If the provided context DOES NOT contain the answer, you must respond EXACTLY with: 'This is something where you need to get in contact with us to help you further.' Do not offer partial guesses.
5. Keep your tone encouraging, instructional, and professional, exactly as Charlyn Ooi would speak to a student.

Context:
{context}

Question: {question}

Answer based ONLY on context above:"""


def get_vectorstore():
    """Initializes or loads the local Chroma vector database."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("Loading existing Knowledge Base database...")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        return vectorstore
        
    print("Database not found. Building Knowledge Base from transcribed Markdown files...")
    
    # Check if there's anything to load
    if not os.path.exists(KB_DIR) or not os.listdir(KB_DIR):
        raise FileNotFoundError(f"No transcribed files found in {KB_DIR}. Run transcribe_videos.py first.")

    # Load all markdown files recursively
    documents = []
    # We iteraterecursive to find everything in knowledge_base subfolders
    for filepath in Path(KB_DIR).rglob('*.md'):
        loader = TextLoader(str(filepath), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
         raise ValueError(f"Found {KB_DIR} folder, but no .md files inside it.")

    print(f"Loaded {len(documents)} document(s). Chunking into pieces...")
    
    # Split the documents into manageable chunks (e.g. 1000 characters, ~250 tokens)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks. Generating embeddings and storing in local Vector DB...")
    
    # Create the vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"Knowledge Base successfully built and saved to {DB_DIR}/ ")
    return vectorstore


def setup_rag_chain():
    """Sets up the Retrieval-Augmented Generation pipeline."""
    vectorstore = get_vectorstore()
    
    # Use the vectorstore as a retriever (fetch top 4 most relevant chunks)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # Define the LLM and the Prompt Template
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.0) # Temperature 0 for maximum strictness/precision
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    # Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # The Chain: Retrieve -> Format -> Inject to Prompt -> LLM Generation
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain


def chat_interface():
    """A simple terminal interface to talk to the chatbot."""
    print("\n========================================================")
    print("Charlyn Ooi Chatbot (Powered by strict RAG)")
    print("Type 'quit' or 'exit' to close the session.")
    print("========================================================\n")
    
    try:
        chain = setup_rag_chain()
    except Exception as e:
        print(f"\nFailed to start chatbot: {e}")
        return

    print("\nInitialization complete! Ask the bot a question about the courses.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Session ended.")
            break
            
        if not user_input.strip():
            continue
            
        print("Bot is thinking...\n")
        response = chain.invoke(user_input)
        print(f"Charlyn Bot: {response}\n")
        print("-" * 50)


if __name__ == "__main__":
    chat_interface()
