import logging
from transformers import logging as transformers_logging
import warnings
from dotenv import load_dotenv  # Make sure this is imported
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load environment variables from .env file
load_dotenv()

# Set log levels
transformers_logging.get_logger("langchain.text_splitter").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Retrieve OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Make sure your .env file has OPENAI_API_KEY set.")

openai.api_key = api_key

# Read contents of Selected_Document.txt into text variable
with open("Selected_Document.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Define variables
chunk_size = 500
chunk_overlap = 100
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 5

# Split text into chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

chunks = text_splitter.split_text(text)

# Load model and encode chunks
embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=False)
embeddings = np.array(embeddings).astype('float32')

# Initialize FAISS index and add embeddings
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Function to retrieve top k chunks for a question
def retrieve_chunks(question: str, k: int = top_k):
    """
    Encode the question and search the FAISS index for top k similar chunks.

    Args:
        question (str): The input question string.
        k (int): Number of nearest chunks to retrieve (default: top_k).

    Returns:
        List[str]: List of relevant text chunks.
    """
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.array(q_vec).astype('float32')
    distances, I = faiss_index.search(q_arr, k)
    return [chunks[i] for i in I[0]]

# Function to answer a question based on retrieved context chunks
def answer_question(question: str) -> str:
    """
    Retrieves relevant chunks and uses OpenAI's Chat Completions API to answer the question.

    Args:
        question (str): The input question string.

    Returns:
        str: The assistant's answer based on the retrieved context.
    """
    # Retrieve relevant chunks
    relevant_chunks = retrieve_chunks(question)

    # Combine chunks into a single context string separated by double newlines
    context = "\n\n".join(relevant_chunks)

    # System prompt defining the assistant's behavior
    system_prompt = (
        "You are a knowledgeable assistant that answers questions based on the provided context. "
        "If the answer is not in the context, say you don’t know."
    )

    # User prompt including the context and the question
    user_prompt = f"""Context:
{context}

Question: {question}

Answer:
"""

    # Call OpenAI Chat Completions with the prompts and parameters
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    # Return the assistant's reply text, stripped of whitespace
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question))
