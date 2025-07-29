
import logging
from transformers import logging as hf_logging
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline

# ─── Suppress noisy logs ────────────────────────────────────────────────────────
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ─── Parameters ────────────────────────────────────────────────────────────────
chunk_size    = 1000
chunk_overlap = 500
model_name    = "sentence-transformers/all-distilroberta-v1"
top_k         = 5

# ─── Read the pre‑scraped document ─────────────────────────────────────────────
with open("Selected_Document.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ─── Split into appropriately‑sized chunks ────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
chunks = splitter.split_text(text)

# ─── Embed & build FAISS index ─────────────────────────────────────────────────
embed_model = SentenceTransformer(model_name)
embeddings  = embed_model.encode(chunks, show_progress_bar=False)
emb_array   = np.array(embeddings, dtype="float32")
index       = faiss.IndexFlatL2(emb_array.shape[1])
index.add(emb_array)

# ─── Load the generator pipeline ───────────────────────────────────────────────
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

# ─── Retrieval & Answering Functions ──────────────────────────────────────────
def retrieve_chunks(question, k=top_k):
    q_emb   = embed_model.encode([question], show_progress_bar=False)
    q_arr   = np.array(q_emb, dtype="float32")
    _, idxs = index.search(q_arr, k)
    return [chunks[i] for i in idxs[0]]

def answer_question(question):
    context = "\n\n".join(retrieve_chunks(question))
    prompt  = (
        "Use the following context to answer the question:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    result = generator(prompt, max_length=200)
    return result[0]["generated_text"]

# ─── Interactive loop ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("\nYour question: ")
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        answer = answer_question(question)
        print("\nAnswer:", answer)
