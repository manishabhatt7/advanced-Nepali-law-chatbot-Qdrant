# advanced-Nepali-law-chatbot-Qdrant
This project aims to make the Constitution of Nepal more accessible through an intuitive RAG chatbot. By combining powerful LLMs (OpenAI), vector search (Qdrant), and intelligent text processing (LangChain), it allows anyone to ask questions and get direct, context-aware answers from the official document. This version includes features such as chat history for conversational context, metadata (page number) sourcing for answer validation, and robust handling of out-of-scope questions.
Below are some of the console outputs.
![Screenshot 2025-05-21 000728](https://github.com/user-attachments/assets/2ac4cc81-4dd7-4fb7-b20a-b33c0d75e834)
![Screenshot 2025-05-21 000713](https://github.com/user-attachments/assets/b0e2ef10-c330-4b2d-ab88-69036810a591)


## Features

*   **Natural Language Queries:** Ask questions about the Constitution of Nepal in plain English.
*   **Retrieval Augmented Generation (RAG):** Answers are generated based on relevant excerpts retrieved from the document.
*   **Qdrant Vector Database:** Utilizes a local Qdrant instance for efficient storage and retrieval of text embeddings.
*   **OpenAI LLMs:** Leverages models like GPT-4o for understanding questions and generating answers.
*   **LangChain Framework:** Orchestrates the entire pipeline, from data loading to conversational interaction.
*   **Source Referencing:** Provides page numbers from the PDF document for cited information, enhancing answer verifiability.
*   **Chat History:** Remembers previous turns in the conversation for more natural follow-up questions.
*   **Out-of-Scope Handling:** Engineered to clearly state when information is not found within the Constitution, minimizing hallucinations.
*   **Metadata Aware:** Extracts and utilizes page numbers during the PDF processing stage.
*   **Error Handling:** Includes basic error handling for a more robust user experience.

## Technologies Used

*   **Python 3.x**
*   **LangChain:** Core framework for building LLM applications.
*   **OpenAI API:** For accessing GPT models (e.g., GPT-4o for generation, text embedding models).
*   **Qdrant:** Vector database for storing and searching document embeddings.
*   **PyMuPDF (fitz):** For extracting text and metadata from PDF documents.
*   **python-dotenv:** For managing environment variables (like API keys).
*   **Docker:** For running the Qdrant instance locally.


## Project Structure (Illustrative)

## Setup and Installation

├── main.py # Main Python script for the chatbot
├── Constitution-of-Nepal.pdf # The source document (ensure you have this)
├── .env # For storing API keys (hidden)
├── .gitignore # Specifies intentionally untracked files by Git
├── requirements.txt # Python package dependencies
└── README.md # This file

### Prerequisites

*   Python 3.8 or higher.
*   `pip` (Python package installer).
*   Docker Desktop installed and running (for Qdrant).
*   An OpenAI API Key.

### 1. Clone the Repository
git clone [Link to your GitHub Repository]
cd [repository-name]

### 2.  Create virtual environment
python -m venv venv

# Activate virtual environment
It's highly recommended to use a virtual environment to manage project dependencies.

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

### 3. Install Dependencies
# Install the required Python packages using the requirements.txt file (create this file first if you haven't - see step below).
pip install -r requirements.txt

# If you don't have requirements.txt yet, you can install packages individually and then generate it:
pip install langchain langchain-openai langchain-community pymupdf qdrant-client openai python-dotenv
pip freeze > requirements.txt

### 4. Set Up Qdrant Locally with Docker
 (Install docker on your desktop first if you don't have already)
# Open a new terminal window and run the Qdrant Docker container:
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

This will download the Qdrant image (if not already present) and start the Qdrant server. Port 6333 is for the gRPC API (used by the Python client) and 6334 is for the HTTP API (Web UI accessible at http://localhost:6334/dashboard). Leave this terminal running.

### 5. Set Up Environment Variables
# Create a .env file in the root directory of the project:
touch .env

#Open the .env file and add your OpenAI API key:
OPENAI_API_KEY="your_openai_api_key_here"
 (Or you can use your gemini key which comes with no cost)

# Important: Ensure .env is listed in your .gitignore file to prevent your API key from being committed to GitHub.

### 6. Place the PDF Document
# Ensure the Constitution-of-Nepal.pdf file is present in the root directory of the project, or update the PDF_PATH variable in the Python script if it's located elsewhere.
# Running the Chatbot
# Once all the setup steps are complete:
# Ensure your Qdrant Docker container is running (from Setup Step 4).
# Ensure your Python virtual environment is activated.
# Run the main Python script from your terminal:
python your_script_name.py

(Replace your_script_name.py with the actual name of your Python file, e.g., app.py or chatbot.py)

# The script will first process the PDF, create embeddings, and load them into Qdrant. This might take a few moments the first time. After initialization, you'll see a prompt to ask questions.
Type "exit" or "quit" to end the chat session.

#### How It Works (Brief Overview)

# 1. Data Loading & Preprocessing:
Text and page number metadata are extracted from Constitution-of-Nepal.pdf using PyMuPDF.
The extracted text (as LangChain Document objects) is split into smaller, manageable chunks.

# 2. Embedding & Indexing:
Each chunk is converted into a numerical vector (embedding) using OpenAI's embedding models.
These embeddings, along with their corresponding text and metadata, are stored in a Qdrant collection.

# 3. Conversational Retrieval:
When a user asks a question:
The question (and chat history) is processed. ConversationalRetrievalChain may condense the question.
The question is embedded, and Qdrant is queried to find the most semantically similar document chunks (the "context").
The original question and the retrieved context are passed to an OpenAI LLM (e.g., GPT-4o) along with carefully engineered prompts.
The LLM generates an answer based only on the provided context.
Chat history is maintained to allow for follow-up questions.

# Future Enhancements (Ideas)

-Integration with a web interface (e.g., Streamlit, Gradio, Flask).
-Support for more advanced retrieval strategies (e.g., MMR, HyDE).
-More sophisticated chunking strategies.
-Automated evaluation pipeline for RAG metrics.
-Ability to upload and process different documents.
-Implementing MCP(Model Context Protocol) with RAG.

#### Contributing
Contributions, issues, and feature requests are welcome!
Please feel free to fork the repository, make changes, and open a pull request. If you plan to make significant changes, please open an issue first to discuss what you would like to change.
