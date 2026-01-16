import json
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# =====================
# CONFIG
# =====================
JSONL_FILES = [
    ("subacar_allFAQ_chunks.jsonl", "faq"),
    ("subacar_chunksV2.jsonl", "package"),
]

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "subacar_all"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# or: "sentence-transformers/all-MiniLM-L6-v2"

# =====================
# Embedding
# =====================
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# =====================
# Load JSONL → Documents
# =====================
documents = []

for path, category in JSONL_FILES:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            # 🔑 เลือก field ที่เป็นข้อความ
            text = (
                obj.get("page_content")
                or obj.get("content")
                or obj.get("text")
            )

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "category": category,
                        "source": path
                    }
                )
            )

print(f"Loaded {len(documents)} documents")

# =====================
# (Optional) Split
# =====================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)

splits = text_splitter.split_documents(documents)
print(f"Split into {len(splits)} chunks")

# =====================
# Build Chroma
# =====================
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME
)

vectorstore.persist()

print("✅ Chroma rebuilt from JSONL")
print(f"📐 Embedding dim = {len(embedding.embed_query('test'))}")