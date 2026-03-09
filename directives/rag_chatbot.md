# Directive: Retrieval-Augmented Generation (RAG) and Chatbot Logic

## Goal
To build a system that searches the transcribed AI knowledge base and generates an informed, accurate, and restricted response based purely on Charlyn Ooi's training material. 

## Inputs
1. **The Knowledge Base**: Processed Markdown files of transcriptions located in `knowledge_base/`.
2. **Milestone Guide**: `N:\1.Current Businesses (S3)\Program Notes\Guide\Table 1-Grid view.csv` (used as contextual rules for progression queries).
3. **User Query**: The question asked by the user in the UI.

## Processing Steps

**Step 1: Chunking & Indexing (Database Setup)**
- Read all `.md` files in the `knowledge_base` directory.
- Split the documents into manageable chunks (e.g., 500-1000 tokens) to preserve context but not exceed LLM context limits.
- Convert each chunk into an embedding using an embedding model like OpenAI's `text-embedding-3-small`.
- Store the embeddings and metadata in a simple, local Vector Database like `ChromaDB` (or an in-memory substitute for MVP) and persist it locally in a `db/` folder.

**Step 2: Semantic Search (Retrieval)**
- Read the incoming User Query and embed it.
- Query the Vector Database to find the top 5 most semantically similar chunks based on cosine similarity logic.

**Step 3: Response Generation (Augmented Generation)**
- Assemble the retrieved chunks into a single "Context Block".
- Pre-pend the Context Block with the following strict System Prompt:

> **System Prompt Guardrails**: 
> "You are an AI assistant for Charlyn Ooi's business and marketing programs (Accelerator Program, Momentum Club, and Leads on Autopilot). Answer the user's question using ONLY the provided context excerpts from the training material. Follow these rules implicitly:
> 1. Do NOT guess, hallucinate, or rely on outside internet knowledge.
> 2. Do NOT invent strategies that are not in the text.
> 3. If the context does NOT contain the answer to the user's question, you must respond EXACTLY with: 'This is something where you need to get in contact with us to help you further.' 
> 4. Keep a helpful, instructional, and encouraging tone."

- Pass the System Prompt, Context Block, and User Query to OpenAI's GPT-4o-mini (or GPT-4o) model to generate the final response.

## Outputs
- The final response string displayed to the user interface.
- If no answer is found, the exact fallback message referring them to the Charlyn Ooi support team.

## Error Handling
- **Database Missing**: If the Vector DB has not been built, the script should automatically build it by scanning `knowledge_base/`.
- **Empty Retrieval**: If the similarity search returns a very low core value, trigger the fallback email message instantly before wasting LLM tokens.
