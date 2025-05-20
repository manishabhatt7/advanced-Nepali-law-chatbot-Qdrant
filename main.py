import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant # Remains the same
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain # Changed for history
from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qdrant_exceptions # For Qdrant specific errors
from openai import APIError, AuthenticationError, APIConnectionError # For OpenAI specific errors
from langchain.prompts import PromptTemplate 


from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os
from uuid import uuid4
import traceback
import logging

# --- LANGCHAIN MESSAGE TYPES FOR HISTORY ---
from langchain_core.messages import HumanMessage, AIMessage

# ------------------------ LOGGING SETUP ------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress overly verbose logs from underlying HTTP libraries if necessary
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# ---------------------------------------------------------------

# ------------------------ LOAD ENV ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = "Constitution-of-Nepal.pdf"

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    exit()
# ----------------------------------------------------------

# Step 1: Extract text and metadata (page numbers) from the PDF
def extract_pages_with_metadata(pdf_path):
    logging.info(f"Extracting text and metadata from PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF {pdf_path}: {e}")
        raise ValueError(f"Could not open or read PDF: {pdf_path}. Ensure it's a valid PDF file.") from e

    documents = []
    if doc.page_count == 0:
        logging.warning("PDF has no pages.")
        raise ValueError("PDF appears to be empty (0 pages) or unreadable!")

    for page_num, page in enumerate(doc):
        text = page.get_text("text", sort=True) # sort=True helps with reading order
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page_number": page_num + 1}))
        else:
            logging.info(f"Page {page_num + 1} has no text content.")
    doc.close()

    if not documents:
        logging.warning("No text could be extracted from the PDF.")
        raise ValueError("No text content found in the PDF!")
    logging.info(f"Extracted {len(documents)} pages with text content.")
    return documents


# Step 2: Split documents into chunks, preserving metadata
def split_documents_into_chunks(documents):
    logging.info("Splitting documents into manageable chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents) # This method preserves metadata
    logging.info(f"Split documents into {len(chunks)} chunks.")
    if chunks:
        logging.debug(f"First chunk preview: Content='{chunks[0].page_content[:50]}...', Metadata={chunks[0].metadata}")
    return chunks


# Step 3: Create Qdrant vector store from chunks (now Document objects)
# Step 3: Create Qdrant vector store from chunks (now Document objects)
def create_qdrant_vector_store(document_chunks, openai_api_key_param):
    logging.info("Creating Qdrant vector store...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key_param)
    embedding_dimension = 1536
    logging.info(f"Using OpenAI embedding model: {embeddings.model} with dimension: {embedding_dimension}")

    qdrant_host = "localhost"
    qdrant_port = 6333

    try:
        # Initialize client
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=10)
        # Try a simple operation to check connectivity, e.g., listing collections (even if empty)
        # This will raise an exception if Qdrant is not reachable.
        client.get_collections()
        logging.info(f"Successfully connected to Qdrant at {qdrant_host}:{qdrant_port}.")
    except ConnectionRefusedError as e:
        logging.error(f"Connection refused by Qdrant server at {qdrant_host}:{qdrant_port}. Is Qdrant server running? Error: {e}")
        raise ConnectionError(f"Could not connect to Qdrant at {qdrant_host}:{qdrant_port}. Please ensure it is running.") from e
    except Exception as e: # Catch other potential qdrant_client or network errors during connection
        logging.error(f"Failed to connect to Qdrant or perform initial check at {qdrant_host}:{qdrant_port}. Error: {e}")
        # You might want to check if e is a qdrant specific connection error if one is available
        # For example, older versions might have qdrant_client.openapi.exceptions.ServiceException
        # For now, a general Exception catch for other connection-related issues.
        raise ConnectionError(f"Could not establish a connection or perform initial check with Qdrant at {qdrant_host}:{qdrant_port}.") from e


    collection_name = "nepal_constitution_v2"

    if not document_chunks:
        logging.error("No document chunks provided to create vector store.")
        raise ValueError("Cannot create vector store with no document chunks.")

    logging.info(f"Number of document chunks for embedding: {len(document_chunks)}")
    logging.debug(f"Sample chunk metadata: {document_chunks[0].metadata}, Content: {document_chunks[0].page_content[:70]}...")

    try:
        logging.info(f"Recreating Qdrant collection: {collection_name} with vector size {embedding_dimension}.")
        # Check if collection exists before trying to recreate (optional, recreate_collection usually handles it)
        # collections_response = client.get_collections()
        # existing_collections = [col.name for col in collections_response.collections]
        # if collection_name in existing_collections:
        #     logging.info(f"Collection '{collection_name}' already exists. Recreating.")

        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
        )
    except Exception as e: # Catch Qdrant-specific errors or general errors during collection management
        logging.error(f"Error during Qdrant collection management for '{collection_name}': {e}")
        raise RuntimeError(f"Failed to manage Qdrant collection '{collection_name}'.") from e

    chunk_contents = [doc.page_content for doc in document_chunks]
    logging.info("Embedding document chunks...")
    try:
        vectors = embeddings.embed_documents(chunk_contents)
    except (APIError, AuthenticationError, APIConnectionError) as e:
        logging.error(f"OpenAI API error during embedding: {e}")
        raise RuntimeError("Failed to embed documents due to OpenAI API issue.") from e
    except Exception as e:
        logging.error(f"Unexpected error during embedding: {e}")
        raise

    logging.info(f"Embedded {len(vectors)} chunks. Dimension of first vector: {len(vectors[0]) if vectors else 'N/A'}")
    if vectors and len(vectors[0]) != embedding_dimension:
        logging.warning(f"Actual embedding dimension {len(vectors[0])} MISMATCHES expected {embedding_dimension}!")

    logging.info(f"Upserting {len(document_chunks)} points to Qdrant collection '{collection_name}'...")
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": str(uuid4()),
                    "vector": vector,
                    "payload": {
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                }
                for vector, chunk in zip(vectors, document_chunks)
            ],
            wait=True
        )
        logging.info(f"Upserted {len(document_chunks)} points to Qdrant.")
    except Exception as e: # Catch Qdrant-specific or general errors during upsert
        logging.error(f"Error upserting points to Qdrant: {e}")
        raise RuntimeError("Failed to upsert data into Qdrant.") from e

    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    logging.info("Qdrant vector store created and wrapped successfully.")
    return vector_store


# Step 4: Setup Conversational QA Chain
def setup_conversational_qa_chain(vector_store, openai_api_key_param):
    logging.info("Setting up ConversationalRetrievalChain with custom prompts...") # Updated log message
    llm = ChatOpenAI(
        openai_api_key=openai_api_key_param,
        model="gpt-4o"
    )

    # Condense Question Prompt (for rephrasing question with history)
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    You are an AI assistant for answering questions ONLY about the Constitution of Nepal, based on the provided text from it.
    If the follow-up question is about a topic clearly outside the Constitution of Nepal (e.g., recipes, other countries' laws), indicate that the question is off-topic for this document.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    # QA Prompt (for answering based on retrieved documents)
    qa_template = """You are an AI assistant for answering questions about the Constitution of Nepal.
    Use ONLY the following pieces of context, which are excerpts from the Constitution of Nepal, to answer the question.
    Your answer MUST be based solely on the information found in the provided context.
    If the provided context does not contain the information to answer the question, you MUST explicitly state: "The provided document (Constitution of Nepal) does not contain information to answer this question."
    Do NOT use any external knowledge. Do NOT make up an answer.
    If the question itself is clearly about a topic not covered by a constitution (e.g., recipes, sports results, other countries' specific laws unless mentioned in the Nepal constitution), state that the question is outside the scope of this document.

    Context:
    {context}

    Question: {question}
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    logging.info("ConversationalRetrievalChain ready with custom prompts.")
    return qa_chain

# Step 5: Ask questions with history
def ask_question_with_history(qa_chain, question, chat_history_messages):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    logging.info(f"Asking question: '{question}' with {len(chat_history_messages)//2} previous turns in history.")
    try:
        # qa_chain.invoke expects 'question' and 'chat_history' (list of BaseMessages or (str,str) tuples)
        response = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history_messages
        })
        return response # Contains 'answer' and 'source_documents'
    except (APIError, AuthenticationError, APIConnectionError) as e:
        logging.error(f"OpenAI API error during QA chain invocation: {e}")
        # Depending on the error, you might want to try again or inform the user.
        if isinstance(e, AuthenticationError):
            raise RuntimeError("OpenAI Authentication Failed. Please check your API key and permissions.") from e
        raise RuntimeError("An issue occurred with the OpenAI API while processing your question.") from e
    except Exception as e:
        logging.error(f"Error during QA chain invocation: {e}", exc_info=True)
        raise RuntimeError("An unexpected error occurred while getting the answer.") from e


# Main function
def main():
    logging.info("Starting Constitution Chatbot...")

    try:
        # Step 1 & 2
        documents_with_metadata = extract_pages_with_metadata(PDF_PATH)
        chunks_with_metadata = split_documents_into_chunks(documents_with_metadata)

        # Step 3
        vector_store = create_qdrant_vector_store(chunks_with_metadata, OPENAI_API_KEY)

        # Step 4
        qa_chain = setup_conversational_qa_chain(vector_store, OPENAI_API_KEY)

    except FileNotFoundError:
        logging.error(f"PDF file not found at path: {PDF_PATH}")
        print(f"\nFATAL ERROR: PDF file '{PDF_PATH}' not found. Please check the path.")
        return
    except ValueError as ve: # Catches custom ValueErrors from our functions
        logging.error(f"Initialization Error (ValueError): {ve}")
        print(f"\nFATAL ERROR during setup: {ve}")
        return
    except ConnectionError as ce: # Catches Qdrant connection issues
        logging.error(f"Initialization Error (ConnectionError): {ce}")
        print(f"\nFATAL ERROR during setup: {ce}")
        return
    except RuntimeError as rte: # Catches other fatal setup issues
        logging.error(f"Initialization Error (RuntimeError): {rte}")
        print(f"\nFATAL ERROR during setup: {rte}")
        return
    except Exception as e: # Catch-all for any other unexpected setup errors
        logging.error(f"An unexpected error occurred during setup: {e}", exc_info=True)
        print(f"\nFATAL UNEXPECTED ERROR during setup: {e}")
        return


    chat_history_messages = [] # Stores LangChain BaseMessage objects

    print("\nConstitution Chatbot is ready. Ask any question about the Nepali Constitution.")
    print("(Type 'exit' or 'quit' to quit.)")

    while True:
        try:
            query = input("\nYour Question: ").strip()
            if not query:
                print("Please enter a question.")
                continue
            if query.lower() in ["exit", "quit"]:
                logging.info("User initiated exit.")
                print("Goodbye!")
                break

            response_data = ask_question_with_history(qa_chain, query, chat_history_messages)
            answer = response_data['answer']
            source_docs = response_data.get('source_documents', []) # Use .get for safety

            print(f"\nAnswer:\n{answer}")

            if source_docs:
                print("\nSources (first 100 chars & page number):")
                unique_sources = {} # To show unique sources by page and content start
                for doc in source_docs:
                    page_num = doc.metadata.get('page_number', 'N/A')
                    content_preview = doc.page_content[:70].strip().replace("\n", " ")
                    source_key = (page_num, content_preview)
                    if source_key not in unique_sources:
                        print(f"- Page: {page_num}, Content: '{content_preview}...'")
                        unique_sources[source_key] = True


            # Update chat history
            chat_history_messages.append(HumanMessage(content=query))
            chat_history_messages.append(AIMessage(content=answer))
            # Optional: Limit history size to prevent excessive token usage / context window issues
            # MAX_HISTORY_TURNS = 5
            # if len(chat_history_messages) > MAX_HISTORY_TURNS * 2:
            #     chat_history_messages = chat_history_messages[-(MAX_HISTORY_TURNS * 2):]


        except ValueError as ve: # For empty question string etc. from ask_question_with_history
            print(f"\nInput Error: {ve}")
        except RuntimeError as rte: # For API or other runtime issues from ask_question_with_history
            print(f"\nProcessing Error: {rte}")
        except Exception as e: # Catch-all for unexpected errors during the loop
            print(f"\nAn unexpected error occurred: {e}")
            logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
            # Optionally, decide if the loop should break or continue
            # For robustness, we'll continue unless it's a critical unrecoverable error

if __name__ == "__main__":
    main()