import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import pandas as pd

# Constants setup
load_dotenv()

# Support both local .env and Streamlit Cloud Secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

KB_DIR = "knowledge_base"
DB_DIR = ".chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Load Milestone Guide
try:
    MILESTONE_DF = pd.read_csv(r"N:\1.Current Businesses (S3)\Program Notes\Guide\Table 1-Grid view.csv")
    MILESTONE_TEXT = MILESTONE_DF[['Action Step / Milestone', 'Phase', 'Month', 'Description', 'Notes']].to_markdown(index=False)
except Exception:
    MILESTONE_TEXT = "Milestone guide could not be loaded."

# Strict Guardrail Prompt with Coaching Abilities
SYSTEM_PROMPT = """You are an expert AI business and marketing assistant for Charlyn Ooi's training programs (Accelerator Program, Momentum Club, and Leads on Autopilot). 

Your primary goal is to act as a **supportive, highly interactive coach** for the user. 
- If a user says "I am stuck" or "What do I do next?", you MUST ask them probing questions to determine where they currently are in the program (e.g., "What phase are you currently working on?" or "Have you defined your ICA yet?"). 
- Use the **Milestone Guide Context** below to structure your advice and guide them to their next logical milestone or action step based on their answers.
- BE HIGHLY CREATIVE: You are fully encouraged to safely combine, synthesize, and brainstorm ideas based on the training material. You should proactively write captions, dream up lead-magnet ideas, act as a collaborative thinking partner, and piece together frameworks to fit the user's specific business niche. 
- CONCRETE EXAMPLES: Whenever you explain a framework, suggest a script, or give advice, you MUST provide a concrete SAMPLE SCENARIO or example roleplay dialogue to show them exactly what it looks like in practice. 
- HELP FIRST, CITE SECOND: Your primary job is to GENUINELY HELP the user. Always lead with a substantive response — a clear summary, actionable advice, a worked example, a creative output, or a framework — based on what the user is asking. THEN, at the end of your first response on a new topic, add a brief source pointer: "You can revisit this in the **[Program Name]** → **[Lesson/Module Name]** for the full walkthrough." Never just point to a lesson without actually helping first. After the first mention on a topic, do NOT keep repeating the source — only cite again when the conversation moves to a new subject.

**THE ONLY STRICT RESTRICTION**:
- While you are allowed to be highly creative with brainstorming, copywriting, and ideation, you MUST NEVER invent raw facts, technical tutorials, or core strategies that contradict or sit completely outside Charlyn Ooi's teachings. If the user asks for factual information, a technical walkthrough, or a completely foreign strategy that is NOT contained in the "Training Context" below, tell them exactly: "This is something where you need to get in contact with us to help you further."
- PHYSICAL HARDWARE RECOMMENDATIONS — NO EXTERNAL LINKS: If you recommend any physical hardware product that a user would need to purchase (e.g., microphones, cameras, lighting, tripods, ring lights, or any other equipment), you MUST NOT include links to any third-party website, store, or retailer (such as Amazon, eBay, manufacturer sites, or any URL). You may name the product and describe it, but the user must find and purchase it on their own. This rule applies only to physical, purchasable hardware items — not to software, courses, or digital services.

=================
MILESTONE GUIDE CONTEXT (The user's progression pathway):
""" + MILESTONE_TEXT + """
=================
TRAINING CONTEXT (Excerpts retrieved from Charlyn Ooi's videos):
{context}
=================

Question/Input: {question}

Assistant:"""

@st.cache_resource(show_spinner="Loading Knowledge Base...")
def load_rag_pipeline():
    """Initializes the vectorstore and the RAG logic."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # If DB doesn't exist, build it from scratch
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        st.info("Building Knowledge Base for the very first time... This might take a moment.")
        documents = []
        for filepath in Path(KB_DIR).rglob('*.md'):
            loader = TextLoader(str(filepath), encoding="utf-8")
            documents.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    else:
        # Load existing Vector DB directly
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.7) # Slightly higher temp to allow for brainstorming

    # Inject MessagesPlaceholder for conversational history
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        formatted = []
        for doc in docs:
            source_program = doc.metadata.get('source', '')
            # Extract program and lesson from the file path in metadata
            # The source is usually the file path, e.g. knowledge_base/Accelerator Program/WEEK 4 - ...md
            source_path = doc.metadata.get('source', '')
            parts = source_path.replace('\\', '/').split('/')
            program = parts[-2] if len(parts) >= 2 else 'Unknown Program'
            lesson = parts[-1].replace('.md', '') if parts else 'Unknown Lesson'
            formatted.append(f"[SOURCE: {program} | LESSON: {lesson}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)
        
    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["question"]) | retriever | format_docs
        )
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Wrap with conversational history
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    return conversational_chain

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ---------- STREAMLIT FRONTEND ----------
st.set_page_config(page_title="Charlyn Ooi Chatbot", page_icon="💡", layout="centered")

st.markdown("<h1 style='text-align: center; color: #E75480;'>Charlyn Ooi Navigator Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask anything about the Accelerator Program, Momentum Club, or Leads on Autopilot!</p>", unsafe_allow_html=True)
st.divider()

# Initialize conversational memory for the session but wipe on refresh
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages from this exact session using Streamlit Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about your program..."):
    # Add user question to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot logic
    with st.chat_message("assistant"):
        with st.spinner("Searching training material..."):
            try:
                rag_chain = load_rag_pipeline()
                
                # Pass session_id to maintain history
                response = rag_chain.invoke(
                    {"question": prompt},
                    config={"configurable": {"session_id": "streamlit_session"}}
                )
                
                st.markdown(response)
                # Save to history for the UI this session
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error accessing the Knowledge Base: {e}")

# Sidebar controls
with st.sidebar:
    st.header("Settings & Info")
    st.markdown("This chatbot uses a **Retrieval-Augmented Generation (RAG)** pipeline to securely search Charlyn's transcripts.")
    
    st.markdown("**Data Privacy:**")
    st.info("Chats are completely ephemeral. When you refresh the webpage, the chat history is immediately wiped.")
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
