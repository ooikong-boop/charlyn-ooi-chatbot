"""
Build the ChromaDB vector database from all transcribed knowledge base files.
Run this script manually whenever new transcriptions are added to the knowledge_base/ folder.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

KB_DIR = "knowledge_base"
DB_DIR = ".chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"

def build_database():
    print("========================================")
    print("Building Charlyn Ooi Knowledge Base...")
    print("========================================")

    # Wipe old DB to ensure a clean rebuild
    import shutil
    shutil.rmtree(DB_DIR, ignore_errors=True)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Load all markdown files from all program subfolders
    documents = []
    md_files = list(Path(KB_DIR).rglob('*.md'))
    print(f"\nFound {len(md_files)} transcribed files across all programs.\n")

    for filepath in md_files:
        loader = TextLoader(str(filepath), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        print("ERROR: No documents found. Run transcribe_videos.py first.")
        return

    print(f"Loaded {len(documents)} documents. Chunking into pieces...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks. Generating embeddings and storing to disk...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print(f"\n========================================")
    print(f"Knowledge Base built and saved to {DB_DIR}/")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"========================================")

if __name__ == "__main__":
    build_database()
