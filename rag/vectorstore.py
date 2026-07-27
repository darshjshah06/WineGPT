import chromadb
import pandas as pd

from rag.embeddings import embed

client = chromadb.PersistentClient(path="./chroma")

collection = client.get_or_create_collection(
    name="wine_collection"
)

def build_index():

    df = pd.read_csv("data/wines.csv")

    for idx, row in df.iterrows():

        document = f"""
        Wine: {row['name']}
        Region: {row['region']}
        Grape: {row['grape']}
        Body: {row['body']}
        Acidity: {row['acidity']}
        Notes: {row['notes']}
        """

        collection.add(
            ids=[str(idx)],
            documents=[document],
            embeddings=[embed(document)]
        )

if __name__ == "__main__":
    build_index()
