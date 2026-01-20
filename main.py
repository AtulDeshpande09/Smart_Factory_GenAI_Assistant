# all the imports
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline , BitsAndBytesConfig
from sentence_transformers import SentenceTransformer , CrossEncoder
import torch

class LocalEmbeddings:
    def __init__(self, model_path="./models/embeddings"):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

embeddings = LocalEmbeddings("./models/embeddings")
reranker = CrossEncoder("./models/cross_enc")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 4-bit quantization
    bnb_4bit_use_double_quant=True,   # double quantization for extra compression
    bnb_4bit_quant_type="nf4",        # better precision scheme
    bnb_4bit_compute_dtype=torch.float16
)

# Loading model
model_name = "./models/phi_mini" 

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",                # automatically assign GPU/CPU
    quantization_config=bnb_config,
)

def rerank(query,docs):
    pairs = [(query, doc.page_content) doc, _ in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs,scores) , key = lambda x: x[1] , reverse=True)
    return [d for (d,s) in reranked]

# load vector DB

CHROMA_PATH = "chroma"

db = Chroma(persist_directory = CHROMA_PATH,
            embedding_function= embeddings)

if __name__ == '__main__':


    query_text = input("Enter You Query : ")

    results = db.similarity_search_with_score(query_text, k=3)
    reranked = rerank(query_text, results)

    PROMPT_TEMPLATE = PROMPT_TEMPLATE = """
        You are an assistant that answers questions strictly using the provided context.

        Rules:
        - Use ONLY the information in the context.
        - Do NOT use prior knowledge.
        - If the answer is not explicitly stated, respond with: "No root cause found in available documents.".
        - Be concise and factual.

        Context:
        {context}

        ---
        Question:
        {query}

        Answer:
    """


    # Actual Prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc,_score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, query = query_text)

    inputs = tokenizer(prompt, return_tensorts="pt")

    with torch.no_grad():
        output_tokens = model.generate(
            **input,
            max_new_tokens = 300,
            temperature = 0.3 ,
            top_p = 0.9 ,
            repetition_penalty = 1.1
        )

    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    sources = [doc.metadata.get("source", None) for doc, _ in results]
    final_response = f"Response:\n{response}"
    print(final_response)
    print()
    print("\Sources : ")
    for doc,_ in results :
        print("-", doc.metadata.get("source, Unknown"))
