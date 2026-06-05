import chromadb
from sentence_transformers import SentenceTransformer
import uuid

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_chroma_client(save_dir: str = "chroma_index"):
    client = chromadb.PersistentClient(path=save_dir)
    return client

def build_faiss_index(chunks: list, save_dir: str = "chroma_index"):
    client = get_chroma_client(save_dir)
    
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

    texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
    embeddings = model.encode(texts).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    metadatas = []
    for c in chunks:
        if isinstance(c, dict):
            metadatas.append({
                "page": str(c.get("page", "?")),
                "heading": str(c.get("heading", "Unknown"))
            })
        else:
            metadatas.append({"page": "?", "heading": "Unknown"})

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Done! Indexed {len(chunks)} chunks in ChromaDB.")
    return collection, chunks

def load_faiss_index(save_dir: str = "chroma_index"):
    client = get_chroma_client(save_dir)
    collection = client.get_or_create_collection(name="documents")
    
    results = collection.get(include=["documents", "metadatas"])
    
    chunks = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({
            "text": doc,
            "page": meta.get("page", "?"),
            "heading": meta.get("heading", "Unknown")
        })
    
    return collection, chunks

def retrieve_top_chunks(query: str, index, chunks: list, top_k: int = 3):
    query_vector = model.encode([query]).tolist()
    
    results = index.query(
        query_embeddings=query_vector,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    top_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        top_chunks.append({
            "text": doc,
            "page": meta.get("page", "?"),
            "heading": meta.get("heading", "Unknown"),
            "score": float(dist)
        })

    return top_chunks