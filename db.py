import json
import os
from langchain.schema import Document
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

DATAPATH = "data/"
DBPATH = "chroma"

# ================================
# Embedding Function
# ================================

class LocalEmbeddings:
    def __init__(self, model_path="./models/embeddings"):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

embeddings = LocalEmbeddings("./models/embeddings")

# ================================
# Load JSONL Documents
# ================================

def load_jsonl_documents(data_path=DATAPATH):
    documents = []

    for filename in os.listdir(data_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(data_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)

                    doc = Document(
                        page_content=record["text"],
                        metadata={
                            **record.get("metadata", {}),
                            "document_type": record.get("document_type"),
                            "source_file": filename,
                            "id": record.get("id")
                        }
                    )
                    documents.append(doc)

    return documents

# ================================
# (Optional) No Chunking
# ================================

def split_text(documents):
    """
    We keep each JSONL record as one chunk.
    This function exists only to keep the pipeline compatible.
    """
    return documents

# ================================
# Main
# ================================

if __name__ == "__main__":

    documents = load_jsonl_documents()
    print(f"Loaded {len(documents)} documents from {DATAPATH}")

    chunks = split_text(documents)
    print("Documents prepared for embedding")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DBPATH
    )

    print(f"Vector DB created and saved at {DBPATH}")
