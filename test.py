from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from datetime import datetime

DBPATH = "chroma"
OUTFILE = "retrieval_results.txt"

# ================================
# Embeddings (same as ingestion)
# ================================

class LocalEmbeddings:
    def __init__(self, model_path="./models/embeddings"):
        self.model = SentenceTransformer(model_path)

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

embeddings = LocalEmbeddings("./models/embeddings")

# ================================
# Load Vector DB
# ================================

db = Chroma(
    persist_directory=DBPATH,
    embedding_function=embeddings
)

# ================================
# Retrieval Tests
# ================================

tests = [
    {
        "name": "Overheating root cause (CNC-02)",
        "query": "Why did CNC-02 overheat?",
        "filter": {"machine_id": "CNC-02"}
    },
    {
        "name": "SOP for overheating",
        "query": "How do I fix coolant overheating?",
        "filter": {"document_type": "sop"}
    },
    {
        "name": "Production impact of overheating",
        "query": "Did overheating affect production?",
        "filter": {"document_type": "production_report"}
    },
    {
        "name": "Controller-related incidents",
        "query": "Controller failure incidents",
        "filter": {"incident_type": "controller"}
    }
]

# ================================
# Run Retrieval + Save Output
# ================================

with open(OUTFILE, "w", encoding="utf-8") as f:
    f.write("RETRIEVAL RESULTS\n")
    f.write(f"Timestamp: {datetime.now()}\n")
    f.write("=" * 100 + "\n\n")

    for i, test in enumerate(tests, start=1):
        f.write(f"TEST {i}: {test['name']}\n")
        f.write(f"Query  : {test['query']}\n")
        f.write(f"Filter : {test['filter']}\n")
        f.write("-" * 100 + "\n")

        results = db.similarity_search(
            test["query"],
            k=3,
            filter=test["filter"]
        )

        if not results:
            f.write("❌ No results found\n\n")
            continue

        for j, doc in enumerate(results, start=1):
            f.write(f"\nResult {j}\n")
            f.write("Text:\n")
            f.write(doc.page_content + "\n\n")
            f.write("Metadata:\n")
            for k, v in doc.metadata.items():
                f.write(f"  {k}: {v}\n")

        f.write("\n" + "=" * 100 + "\n\n")

print(f"✅ Retrieval complete. Results saved to '{OUTFILE}'")
