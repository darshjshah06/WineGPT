from rag.embeddings import embed
from rag.vectorstore import collection

def retrieve(query: str, k: int = 3):

    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=k
    )

    return results["documents"][0]